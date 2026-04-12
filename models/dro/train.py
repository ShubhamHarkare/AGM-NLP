#DESC: This file contains the training and evaluation loop for Group DRO over RoBERTa.
#      Group DRO (Sagawa et al., 2020) minimizes worst-case group loss across source domains.
#      Each source domain is treated as a group, and group weights are updated online
#      to upweight domains with higher loss.

import os
import wandb
import sys
import torch
import itertools
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.roberta.roberta_model import RobertaClassifier
from models.dataset import SentimentDataset

import numpy as np

load_dotenv()

DOMAINS    = ['imdb', 'amazon', 'hotel', 'sentiment']
DATA_ROOT  = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
BATCH_SIZE = 32
MAX_LENGTH = 256
EPOCHS     = 50
LR         = 2e-5
WARMUP_RATIO = 0.1
SEEDS      = [42, 43, 44, 45, 46, 47, 48, 49]
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Group DRO hyperparameters ────────────────────────────────────────
# Step size for the online group weight update (η in the paper).
# Sagawa et al. recommend 0.01 as a default; we can tune if needed.
DRO_ETA = 0.01

# Strong L2 regularization is critical for Group DRO on overparameterized
# models (Section 4 of Sagawa et al., 2020).
WEIGHT_DECAY = 0.01

# Group adjustment constant (C in the paper). 
# Adjusts for imbalanced group sizes. Typically tuned between 0.1 and 3.0.
GROUP_ADJ_C = 1.5 
# ─────────────────────────────────────────────────────────────────────


def get_domain_dataloaders(target_domain, split, tokenizer, shuffle=False):
    """
    Returns a list of DataLoaders, one per source domain (excluding target).
    This is similar to how IRM handles per-domain batching.
    """
    dataloaders = []
    domain_names = []
    for domain in DOMAINS:
        if domain == target_domain:
            continue
        path = os.path.join(DATA_ROOT, domain, split)
        dataset = SentimentDataset(path, tokenizer, max_length=MAX_LENGTH)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
        dataloaders.append(loader)
        domain_names.append(domain)
    return dataloaders, domain_names


def get_target_dataloader(domain, split, tokenizer, shuffle=False):
    """Single dataloader for the held-out target domain."""
    path = os.path.join(DATA_ROOT, domain, split)
    dataset = SentimentDataset(path, tokenizer, max_length=MAX_LENGTH)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )


def evaluate(model, dataloader):
    """Evaluate model on a single dataloader. Returns (F1, accuracy)."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            token_type_ids = batch.get('token_type_ids', None)

            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)

            logits = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if len(all_labels) == 0:
        return 0.0, 0.0

    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc


def evaluate_combined_source(model, dataloaders):
    """Evaluate model across all source domain dataloaders combined."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for loader in dataloaders:
            for batch in loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                token_type_ids = batch.get('token_type_ids', None)

                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(DEVICE)

                logits = model(input_ids, attention_mask, token_type_ids)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    if len(all_labels) == 0:
        return 0.0, 0.0

    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc


def train(target_domain, seed, tokenizer):
    """
    Group DRO training loop.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    run = wandb.init(
        project='AGM-NLP-Research',
        name=f'dro_{target_domain}_seed{seed}',
        config={
            'model': 'dro-roberta',
            'target_domain': target_domain,
            'seed': seed,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'max_length': MAX_LENGTH,
            'epochs': EPOCHS,
            'dro_eta': DRO_ETA,
            'weight_decay': WEIGHT_DECAY,
            'group_adj_c': GROUP_ADJ_C
        }
    )

    # Get per-domain dataloaders for training and validation
    train_loaders, domain_names = get_domain_dataloaders(
        target_domain, 'train', tokenizer, shuffle=True
    )
    val_loaders, _ = get_domain_dataloaders(
        target_domain, 'val', tokenizer, shuffle=False
    )

    num_groups = len(train_loaders)
    print(f"Training on {num_groups} source domains: {domain_names}")

    # Compute group adjustments based on dataset sizes (C / sqrt(n_g))
    group_sizes = torch.tensor([len(loader.dataset) for loader in train_loaders], dtype=torch.float32, device=DEVICE)
    group_adjustments = GROUP_ADJ_C / torch.sqrt(group_sizes)

    model = RobertaClassifier(num_labels=2).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Use the longest loader to define the epoch length to avoid dropping data
    steps_per_epoch = max(len(loader) for loader in train_loaders)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # ── Group DRO weights ──
    group_weights = torch.ones(num_groups, device=DEVICE) / num_groups

    # Early stopping based on worst-case validation
    best_worst_val_f1 = 0.0
    best_state = None
    patience = 3
    patience_count = 0

    # Create infinite iterators for training loaders
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
                
    train_iterators = [iter(cycle(loader)) for loader in train_loaders]

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        epoch_group_losses = [0.0] * num_groups

        for step in range(steps_per_epoch):
            # Fetch one batch from each domain iterator
            batches = [next(it) for it in train_iterators]
            optimizer.zero_grad()

            group_losses = []

            for g, batch in enumerate(batches):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                token_type_ids = batch.get('token_type_ids', None)

                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(DEVICE)

                logits = model(input_ids, attention_mask, token_type_ids)
                per_sample_loss = criterion(logits, labels)
                group_loss = per_sample_loss.mean()
                group_losses.append(group_loss)
                epoch_group_losses[g] += group_loss.item()

            group_losses_tensor = torch.stack(group_losses)

            # ── Online group weight update ──
            with torch.no_grad():
                # Add size penalty adjustment to the loss before exponentiating
                adjusted_losses = group_losses_tensor + group_adjustments
                group_weights = group_weights * torch.exp(DRO_ETA * adjusted_losses)
                group_weights = group_weights / group_weights.sum()  # renormalize

            # Weighted loss uses the unadjusted losses
            total_loss = (group_weights * group_losses_tensor).sum()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            train_loss += total_loss.item()

        avg_train_loss = train_loss / steps_per_epoch

        # Validation: evaluate on each domain separately
        val_f1s = []
        val_accs = []
        for loader in val_loaders:
            f1, acc = evaluate(model, loader)
            val_f1s.append(f1)
            val_accs.append(acc)
            
        worst_val_f1 = min(val_f1s)
        avg_val_f1 = np.mean(val_f1s)

        # Log per-group losses for monitoring
        group_loss_log = {}
        for g, name in enumerate(domain_names):
            group_loss_log[f'group_loss/{name}'] = epoch_group_losses[g] / steps_per_epoch
            group_loss_log[f'group_weight/{name}'] = group_weights[g].item()
            group_loss_log[f'val_f1/{name}'] = val_f1s[g]

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Worst Val F1: {worst_val_f1:.4f} | "
              f"Avg Val F1: {avg_val_f1:.4f} | "
              f"Weights: {[f'{w:.3f}' for w in group_weights.tolist()]}")

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'worst_val_f1': worst_val_f1,
            'avg_val_f1': avg_val_f1,
            **group_loss_log,
        })

        # Early stopping relies strictly on the WORST performing group
        if worst_val_f1 > best_worst_val_f1:
            best_worst_val_f1 = worst_val_f1
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            print(f"No improvement in worst-group F1: {patience_count}/{patience}")

            if patience_count >= patience:
                print(f'Early stopping triggered at epoch: {epoch + 1}')
                break

    wandb.finish()

    # Restore best robust weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    return model


def evaluate_cross_domain(model, target_domain, tokenizer, seed):
    """Evaluate trained model: source F1 (combined), target F1, delta, TE."""
    model.eval()

    # Step 1: evaluate on combined source domains test set
    test_loaders, _ = get_domain_dataloaders(target_domain, 'test', tokenizer, shuffle=False)
    source_f1, source_acc = evaluate_combined_source(model, test_loaders)

    # Step 2: evaluate on held-out target domain test set
    target_loader = get_target_dataloader(target_domain, 'test', tokenizer, shuffle=False)
    target_f1, target_acc = evaluate(model, target_loader)

    # Step 3: compute delta and TE
    delta = abs(source_f1 - target_f1)
    te = target_f1 / source_f1 if source_f1 > 0 else 0.0

    # Step 4: print results
    print(f"  Source (combined 3 domains) F1 : {source_f1:.4f}")
    print(f"  Target ({target_domain}) F1    : {target_f1:.4f}")
    print(f"  Δ: {delta:.4f} | TE: {te:.4f}")

    # Step 5: log to wandb
    wandb.log({
        f'test/source_f1': source_f1,
        f'test/target_f1_{target_domain}': target_f1,
        f'delta/{target_domain}': delta,
        f'te/{target_domain}': te,
        'seed': seed
    })

    return {
        'source_f1': source_f1,
        'target_f1': target_f1,
        'delta': delta,
        'te': te
    }


def main():
    wandb.login(key=os.environ.get('WANDB_API_KEY'))

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    for target in DOMAINS:
        print(f'\n{"="*60}')
        print(f'Group DRO — Target domain: {target}')
        print(f'{"="*60}')

        seed_results = []

        for seed in SEEDS:
            print(f'\nSEED: {seed}')

            model = train(target_domain=target, seed=seed, tokenizer=tokenizer)

            wandb.init(
                project='AGM-NLP-Research',
                name=f'dro_{target}_seed{seed}_eval',
                config={
                    'model': 'dro-roberta',
                    'target_domain': target,
                    'seed': seed
                }
            )

            results = evaluate_cross_domain(model, target, tokenizer, seed)
            seed_results.append(results)
            wandb.finish()

            del model
            torch.cuda.empty_cache()

        # Summary across seeds
        wandb.init(
            project='AGM-NLP-Research',
            name=f'dro_{target}_summary',
            config={
                'model': 'dro-roberta',
                'target_domain': target,
                'seeds': SEEDS,
                'batch_size': BATCH_SIZE,
                'lr': LR,
                'max_length': MAX_LENGTH,
                'epochs': EPOCHS,
                'dro_eta': DRO_ETA,
                'group_adj_c': GROUP_ADJ_C
            }
        )

        print(f"\n--- Summary across seeds for target: {target} ---")

        mean_source_f1 = np.mean([r['source_f1'] for r in seed_results])
        std_source_f1  = np.std([r['source_f1']  for r in seed_results])
        mean_target_f1 = np.mean([r['target_f1'] for r in seed_results])
        std_target_f1  = np.std([r['target_f1']  for r in seed_results])
        mean_delta     = np.mean([r['delta']      for r in seed_results])
        std_delta      = np.std([r['delta']       for r in seed_results])
        mean_te        = np.mean([r['te']         for r in seed_results])
        std_te         = np.std([r['te']          for r in seed_results])

        print(f"  Source (combined) F1 : {mean_source_f1:.4f}±{std_source_f1:.4f}")
        print(f"  Target ({target}) F1 : {mean_target_f1:.4f}±{std_target_f1:.4f}")
        print(f"  Δ: {mean_delta:.4f}±{std_delta:.4f} | TE: {mean_te:.4f}±{std_te:.4f}")

        wandb.log({
            f'summary/mean_source_f1':          mean_source_f1,
            f'summary/std_source_f1':           std_source_f1,
            f'summary/mean_target_f1_{target}': mean_target_f1,
            f'summary/std_target_f1_{target}':  std_target_f1,
            f'summary/mean_delta_{target}':     mean_delta,
            f'summary/std_delta_{target}':      std_delta,
            f'summary/mean_te_{target}':        mean_te,
            f'summary/std_te_{target}':         std_te,
        })

        wandb.finish()
        print(f"\nCompleted all seeds for target domain: {target}")


if __name__ == '__main__':
    main()