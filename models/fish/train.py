
#DESC: This file contains the training and evaluation loop for Fish over RoBERTa.
#      Fish (Shi et al., 2022) aligns inter-domain gradient directions. 
#      It uses a Reptile-style meta-learning inner loop to achieve this.

import os
import wandb
import sys
import torch
import random
import itertools
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# Adjust your paths as necessary
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.roberta.roberta_model import RobertaClassifier
from models.dataset import SentimentDataset

load_dotenv()

DOMAINS    = ['imdb', 'amazon', 'hotel', 'sentiment']
DATA_ROOT  = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
BATCH_SIZE = 32
MAX_LENGTH = 256
EPOCHS     = 50
LR         = 2e-5  # This acts as the Meta-LR for the outer AdamW optimizer
WARMUP_RATIO = 0.1
SEEDS      = [42, 43, 44, 45, 46, 47, 48, 49]
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Fish hyperparameters ─────────────────────────────────────────────
# The learning rate for the inner, sequential SGD steps.
# Typical values for Fish inner LR are between 1e-4 and 1e-3
FISH_INNER_LR = 1e-4
WEIGHT_DECAY = 0.01
# ─────────────────────────────────────────────────────────────────────


def get_domain_dataloaders(target_domain, split, tokenizer, shuffle=False):
    """Returns a list of DataLoaders, one per source domain (excluding target)."""
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
    Fish training loop.
    Uses the proper Reptile-style inner/outer loop to maximize gradient alignment.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run = wandb.init(
        project='AGM-NLP-Research',
        name=f'fish_{target_domain}_seed{seed}',
        config={
            'model': 'fish-roberta',
            'target_domain': target_domain,
            'seed': seed,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'fish_inner_lr': FISH_INNER_LR,
            'max_length': MAX_LENGTH,
            'epochs': EPOCHS,
            'weight_decay': WEIGHT_DECAY,
        }
    )

    # Get per-domain dataloaders
    train_loaders, domain_names = get_domain_dataloaders(
        target_domain, 'train', tokenizer, shuffle=True
    )
    val_loaders, _ = get_domain_dataloaders(
        target_domain, 'val', tokenizer, shuffle=False
    )

    num_groups = len(train_loaders)
    print(f"Training on {num_groups} source domains: {domain_names}")

    model = RobertaClassifier(num_labels=2).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Use max length so we don't drop data from large domains
    steps_per_epoch = max(len(loader) for loader in train_loaders)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Early stopping
    best_val_f1 = 0.0
    best_state = None
    patience = 3
    patience_count = 0

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    
    train_iterators = [iter(cycle(loader)) for loader in train_loaders]

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for step in range(steps_per_epoch):
            batches = [next(it) for it in train_iterators]
            
            # ── Step 1: Save the starting weights for this meta-step ──
            theta_0 = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

            # ── Step 2: Shuffle domain order (Critical for unbiased Fish updates) ──
            random.shuffle(batches)

            step_loss = 0

            # ── Step 3: Inner Loop - Sequential updates ──
            for batch in batches:
                model.zero_grad()
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                token_type_ids = batch.get('token_type_ids', None)

                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(DEVICE)

                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
                loss.backward()
                step_loss += loss.item()

                # Take an inner SGD step directly on the parameters
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.sub_(FISH_INNER_LR * p.grad)
            
            # ── Step 4: Outer Loop - Meta update ──
            optimizer.zero_grad()
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        # The pseudo-gradient is the difference between original and inner-updated weights
                        pseudo_grad = theta_0[n] - p.data
                        
                        # Restore original weights so outer optimizer starts from theta_0
                        p.data.copy_(theta_0[n])
                        
                        # Set gradient for the outer optimizer
                        p.grad = pseudo_grad

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # AdamW takes a step in the direction of the pseudo-gradient
            optimizer.step()
            scheduler.step()
            epoch_loss += (step_loss / num_groups)

        avg_train_loss = epoch_loss / steps_per_epoch

        # Validation
        val_f1, val_acc = evaluate_combined_source(model, val_loaders)

        # Per-domain validation
        val_log = {}
        for g, loader in enumerate(val_loaders):
            domain_f1, _ = evaluate(model, loader)
            val_log[f'val_f1/{domain_names[g]}'] = domain_f1

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_f1': val_f1,
            'val_acc': val_acc,
            **val_log,
        })

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            print(f"No improvement {patience_count}/{patience}")

            if patience_count >= patience:
                print(f'Early stopping triggered at epoch: {epoch + 1}')
                break

    wandb.finish()

    # Restore best weights
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
        print(f'Fish — Target domain: {target}')
        print(f'{"="*60}')

        seed_results = []

        for seed in SEEDS:
            print(f'\nSEED: {seed}')

            model = train(target_domain=target, seed=seed, tokenizer=tokenizer)

            wandb.init(
                project='AGM-NLP-Research',
                name=f'fish_{target}_seed{seed}_eval',
                config={
                    'model': 'fish-roberta',
                    'target_domain': target,
                    'seed': seed
                }
            )

            results = evaluate_cross_domain(model, target, tokenizer, seed)
            seed_results.append(results)
            wandb.finish()

            # Free GPU memory before next seed
            del model
            torch.cuda.empty_cache()

        # Summary across seeds
        wandb.init(
            project='AGM-NLP-Research',
            name=f'fish_{target}_summary',
            config={
                'model': 'fish-roberta',
                'target_domain': target,
                'seeds': SEEDS,
                'batch_size': BATCH_SIZE,
                'lr': LR,
                'max_length': MAX_LENGTH,
                'epochs': EPOCHS,
                'fish_inner_lr': FISH_INNER_LR,
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