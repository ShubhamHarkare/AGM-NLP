# DESC: This file is responsible for training the AGM model
# Variant 2: No Mask / Random Mask Ablation

import os
import sys
import wandb
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from transformers import RobertaTokenizer, RobertaForMaskedLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.agm.model import AGMModel
from models.dataset import SentimentDataset
from models.agm.model import compute_gradient_input, detect_spurious_tokens, generate_counterfactual, filter_counterfactual
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DOMAINS      = ['imdb', 'amazon', 'hotel', 'sentiment']
DATA_ROOT    = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
CKPT_DIR     = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
BATCH_SIZE   = 8    # smaller than other models — AGM is memory intensive
MAX_LENGTH   = 256
EPOCHS       = 10
LR           = 2e-5
WARMUP_RATIO = 0.1
SEEDS        = [42, 43, 44]
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LAMBDA1      = 0.1   # weight for L_mask
LAMBDA2      = 0.1   # weight for L_CCL


# ── Data ──────────────────────────────────────────────────────────────────────

def get_dataloader(target_domain, split, tokenizer, shuffle=False):
    '''
    Leave-one-out dataloader — combines 3 source domains, excludes target.
    No domain ID needed — AGM does not use a domain classifier.
    '''
    datasets = []
    for domain in DOMAINS:
        if domain == target_domain:
            continue
        path = os.path.join(DATA_ROOT, domain, split)
        dataset = SentimentDataset(path, tokenizer, max_length=MAX_LENGTH)
        datasets.append(dataset)

    combined = ConcatDataset(datasets)
    return DataLoader(
        combined,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )


def get_target_dataloader(domain, split, tokenizer, shuffle=False):
    path = os.path.join(DATA_ROOT, domain, split)
    dataset = SentimentDataset(path, tokenizer, max_length=MAX_LENGTH)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, dataloader):
    '''
    Evaluates AGMModel on a dataloader.
    AGMModel returns (logits, pooled_output, last_hidden_state).
    Only logits are needed for evaluation.
    '''
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)

            # AGMModel returns 3 values — only need logits for evaluation
            logits, _, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1  = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc


def evaluate_cross_domain(model, target_domain, tokenizer, seed):
    model.eval()

    # source: combined 3 training domains
    all_preds  = []
    all_labels = []
    train_test_loaders = get_dataloader(target_domain, 'test', tokenizer)

    with torch.no_grad():
        for batch in train_test_loaders.dataset.__iter__() if hasattr(
                train_test_loaders.dataset, '__iter__') else []:
            pass

    # cleaner approach — iterate dataloader directly
    with torch.no_grad():
        for batch in DataLoader(
            train_test_loaders.dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4
        ):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)
            logits, _, _   = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    source_f1  = f1_score(all_labels, all_preds, average='macro')
    source_acc = accuracy_score(all_labels, all_preds)

    # target: held-out domain
    target_loader = get_target_dataloader(target_domain, 'test', tokenizer)
    target_f1, target_acc = evaluate(model, target_loader)

    delta = abs(source_f1 - target_f1)
    te    = target_f1 / source_f1 if source_f1 > 0 else 0.0

    print(f"  Source (combined) F1 : {source_f1:.4f}")
    print(f"  Target ({target_domain}) F1 : {target_f1:.4f}")
    print(f"  Δ: {delta:.4f} | TE: {te:.4f}")

    wandb.log({
        f'test/source_f1':                 source_f1,
        f'test/target_f1_{target_domain}': target_f1,
        f'delta/{target_domain}':          delta,
        f'te/{target_domain}':             te,
        'seed': seed
    })

    return {
        'source_f1': source_f1,
        'target_f1': target_f1,
        'delta':     delta,
        'te':        te
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(target_domain, seed, tokenizer):
    torch.manual_seed(seed)
    np.random.seed(seed)

    run = wandb.init(
        project='AGM-NLP-2', # FIX: Updated Project Name
        name=f'agm_no_mask_{target_domain}_seed{seed}',
        config={
            'model':         'agm-roberta',
            'target_domain': target_domain,
            'seed':          seed,
            'batch_size':    BATCH_SIZE,
            'lr':            LR,
            'max_length':    MAX_LENGTH,
            'epochs':        EPOCHS,
            'lambda1':       LAMBDA1,
            'lambda2':       LAMBDA2,
        }
    )

    train_loader  = get_dataloader(target_domain, 'train', tokenizer, shuffle=True)
    # FIX: Use source domains' validation split to prevent Target Domain Leakage
    val_loader    = get_dataloader(target_domain, 'val', tokenizer)

    # ── Models ────────────────────────────────────────────────────────────────
    model = AGMModel(num_labels=2).to(DEVICE)
    model.roberta.gradient_checkpointing_enable()

    # MLM model for counterfactual generation — shares backbone with AGMModel
    mlm_model = RobertaForMaskedLM.from_pretrained('roberta-base')
    mlm_model.roberta = model.roberta   # share backbone weights
    mlm_model = mlm_model.to(DEVICE)
    mlm_model.eval()

    mask_token_id = tokenizer.mask_token_id

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss()

    # ── Checkpoint & Early Stopping ───────────────────────────────────────────
    run_ckpt_dir = os.path.join(CKPT_DIR, f'agm_no_mask_{target_domain}_seed{seed}')
    os.makedirs(run_ckpt_dir, exist_ok=True)

    best_val_f1    = 0.0
    patience       = 3
    patience_count = 0

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        train_loss      = 0
        total_l_ce      = 0
        total_l_mask    = 0
        total_l_ccl     = 0

        for batch_idx, batch in enumerate(train_loader):

            # ── 1. Move to device ─────────────────────────────────────────────
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)

            # ── 2. Forward pass ───────────────────────────────────────────────
            logits, pooled_output, last_hidden_state = model(
                input_ids, attention_mask
            )

            # ── 4. Compute L_CE ───────────────────────────────────────────────
            L_CE = criterion(logits, labels)
            
            torch.cuda.empty_cache()

            # ── 7. Detect spurious tokens (Randomly) ──────────────────────────
            # FIX: Ensure we only mask real tokens, not <pad> tokens!
            spurious_mask = (torch.rand(input_ids.shape, device=DEVICE) > 0.75) & attention_mask.bool()

            # ── 8. Compute L_mask ─────────────────────────────────────────────
            L_mask = torch.tensor(0.0, device=DEVICE)

            # ── 9. Generate counterfactual x' ─────────────────────────────────
            # FIX: Passed attention_mask to prevent padding token prediction
            counterfactual_ids = generate_counterfactual(
                input_ids, attention_mask, spurious_mask, mlm_model, mask_token_id
            )

            # ── 10. Filter counterfactual — keep only safe ones ───────────────
            valid = filter_counterfactual(
                input_ids, counterfactual_ids, model, attention_mask
            )

            # ── 11. Compute L_CCL ─────────────────────────────────────────────
            if valid.any():
                _, counterfactual_pooled, _ = model(
                    counterfactual_ids[valid],
                    attention_mask[valid]
                )
                L_CCL = F.mse_loss(
                    pooled_output[valid].detach(),
                    counterfactual_pooled
                )
            else:
                L_CCL = torch.tensor(0.0, device=DEVICE)

            torch.cuda.empty_cache()
            
            # ── 12. Combine losses ────────────────────────────────────────────
            optimizer.zero_grad()
            L_AGM = L_CE + LAMBDA2 * L_CCL

            # ── 13. Backward on full AGM loss ─────────────────────────────────
            L_AGM.backward()

            # ── 14. Clip gradients & update ───────────────────────────────────
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss   += L_AGM.item()
            total_l_ce   += L_CE.item()
            total_l_mask += L_mask.item()
            total_l_ccl  += L_CCL.item()

        # ── End of epoch ──────────────────────────────────────────────────────
        avg_loss     = train_loss   / len(train_loader)
        avg_l_ce     = total_l_ce   / len(train_loader)
        avg_l_mask   = total_l_mask / len(train_loader)
        avg_l_ccl    = total_l_ccl  / len(train_loader)

        val_f1, val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"L_CE: {avg_l_ce:.4f} | "
              f"L_mask: {avg_l_mask:.4f} | "
              f"L_CCL: {avg_l_ccl:.4f} | "
              f"Val F1: {val_f1:.4f}")

        wandb.log({
            'epoch':      epoch + 1,
            'train_loss': avg_loss,
            'l_ce':       avg_l_ce,
            'l_mask':     avg_l_mask,
            'l_ccl':      avg_l_ccl,
            'val_f1':     val_f1,
            'val_acc':    val_acc,
        })

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            torch.save(
                model.state_dict(),
                os.path.join(run_ckpt_dir, 'agm_best_model.pt')
            )
            print(f'New best model saved at val_f1: {best_val_f1:.4f}')
        else:
            patience_count += 1
            print(f'No improvement {patience_count}/{patience}')
            if patience_count >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    wandb.finish()
    return os.path.join(run_ckpt_dir, 'agm_best_model.pt')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    wandb.login(key=os.environ.get('WANDB_API_KEY'))

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    for target in ['sentiment']:
        print(f'\nTarget domain: {target}')
        seed_results = []

        for seed in SEEDS:
            print(f'Seed: {seed}')
            ckpt_path = train(target, seed, tokenizer)

            model = AGMModel(num_labels=2).to(DEVICE)
            model.load_state_dict(
                torch.load(ckpt_path, map_location=DEVICE)
            )

            wandb.init(
                project='AGM-NLP-2', # FIX: Updated Project Name
                name=f'agm_no_mask_{target}_seed{seed}_eval',
                config={
                    'model':         'agm-roberta',
                    'target_domain': target,
                    'seed':          seed
                }
            )

            results = evaluate_cross_domain(model, target, tokenizer, seed)
            seed_results.append(results)
            wandb.finish()

        # Summary across seeds
        wandb.init(
            project='AGM-NLP-2', # FIX: Updated Project Name
            name=f'agm_no_mask_{target}_summary',
            config={
                'model':         'agm-roberta',
                'target_domain': target,
                'seeds':         SEEDS,
                'lambda1':       LAMBDA1,
                'lambda2':       LAMBDA2,
            }
        )

        mean_source_f1 = np.mean([r['source_f1'] for r in seed_results])
        std_source_f1  = np.std([r['source_f1']  for r in seed_results])
        mean_target_f1 = np.mean([r['target_f1'] for r in seed_results])
        std_target_f1  = np.std([r['target_f1']  for r in seed_results])
        mean_delta     = np.mean([r['delta']      for r in seed_results])
        std_delta      = np.std([r['delta']       for r in seed_results])
        mean_te        = np.mean([r['te']         for r in seed_results])
        std_te         = np.std([r['te']          for r in seed_results])

        print(f"\n--- Summary for target: {target} ---")
        print(f"  Source F1 : {mean_source_f1:.4f}±{std_source_f1:.4f}")
        print(f"  Target F1 : {mean_target_f1:.4f}±{std_target_f1:.4f}")
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
        print(f"Completed all seeds for target: {target}")


if __name__ == '__main__':
    main()