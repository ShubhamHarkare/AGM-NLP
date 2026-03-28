# DESC: Ablation 1 — No CCL (L_CE + L_mask only)
# Removes counterfactual generation and contrastive loss.
# L_AGM = L_CE + LAMBDA1 * L_mask

import os
import sys
import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.agm.model import AGMModel
from models.dataset import SentimentDataset
from models.agm.model import compute_gradient_input, detect_spurious_tokens

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DOMAINS      = ['imdb', 'amazon', 'hotel', 'sentiment']
DATA_ROOT    = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
CKPT_DIR     = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
BATCH_SIZE   = 8
MAX_LENGTH   = 256
EPOCHS       = 10
LR           = 2e-5
WARMUP_RATIO = 0.1
SEEDS        = [42, 43, 44]
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LAMBDA1      = 0.1   # weight for L_mask


# ── Data ──────────────────────────────────────────────────────────────────────

def get_dataloader(target_domain, split, tokenizer, shuffle=False):
    datasets = []
    for domain in DOMAINS:
        if domain == target_domain:
            continue
        path = os.path.join(DATA_ROOT, domain, split)
        dataset = SentimentDataset(path, tokenizer, max_length=MAX_LENGTH)
        datasets.append(dataset)
    combined = ConcatDataset(datasets)
    return DataLoader(combined, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=4, pin_memory=True)


def get_target_dataloader(domain, split, tokenizer, shuffle=False):
    path = os.path.join(DATA_ROOT, domain, split)
    dataset = SentimentDataset(path, tokenizer, max_length=MAX_LENGTH)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=4, pin_memory=True)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)
            logits, _, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1  = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc


def evaluate_cross_domain(model, target_domain, tokenizer, seed):
    model.eval()
    all_preds, all_labels = [], []
    train_test_loaders = get_dataloader(target_domain, 'test', tokenizer)
    with torch.no_grad():
        for batch in DataLoader(train_test_loaders.dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=4):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)
            logits, _, _   = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    source_f1  = f1_score(all_labels, all_preds, average='macro')
    target_loader = get_target_dataloader(target_domain, 'test', tokenizer)
    target_f1, _ = evaluate(model, target_loader)
    delta = abs(source_f1 - target_f1)
    te    = target_f1 / source_f1 if source_f1 > 0 else 0.0

    print(f"  Source F1: {source_f1:.4f} | Target ({target_domain}) F1: {target_f1:.4f} | Δ: {delta:.4f} | TE: {te:.4f}")
    wandb.log({
        'test/source_f1': source_f1, f'test/target_f1_{target_domain}': target_f1,
        f'delta/{target_domain}': delta, f'te/{target_domain}': te, 'seed': seed
    })
    return {'source_f1': source_f1, 'target_f1': target_f1, 'delta': delta, 'te': te}


# ── Training ──────────────────────────────────────────────────────────────────

def train(target_domain, seed, tokenizer):
    torch.manual_seed(seed)
    np.random.seed(seed)

    run = wandb.init(
        project='AGM-NLP-2',
        name=f'ablation1_noccl_{target_domain}_seed{seed}',
        config={
            'model': 'agm-ablation1-noccl', 'target_domain': target_domain,
            'seed': seed, 'batch_size': BATCH_SIZE, 'lr': LR,
            'max_length': MAX_LENGTH, 'epochs': EPOCHS, 'lambda1': LAMBDA1,
        }
    )

    train_loader = get_dataloader(target_domain, 'train', tokenizer, shuffle=True)
    val_loader   = get_dataloader(target_domain, 'val', tokenizer)

    model = AGMModel(num_labels=2).to(DEVICE)

    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion    = torch.nn.CrossEntropyLoss()

    run_ckpt_dir = os.path.join(CKPT_DIR, f'ablation1_noccl_{target_domain}_seed{seed}')
    os.makedirs(run_ckpt_dir, exist_ok=True)
    best_val_f1, patience, patience_count = 0.0, 3, 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, total_l_ce, total_l_mask = 0, 0, 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)

            # Forward
            logits, pooled_output, last_hidden_state = model(input_ids, attention_mask)

            # L_CE
            L_CE = criterion(logits, labels)

            # Double backprop for attribution
            grads = torch.autograd.grad(
                outputs=L_CE, inputs=last_hidden_state,
                create_graph=True, retain_graph=True
            )[0]

            torch.cuda.empty_cache()

            # Attribution + spurious detection
            attribution   = compute_gradient_input(last_hidden_state, grads)
            spurious_mask = detect_spurious_tokens(attribution, tau_high=0.75)

            # L_mask
            if spurious_mask.any():
                L_mask = (attribution[spurious_mask] ** 2).mean()
            else:
                L_mask = torch.tensor(0.0, device=DEVICE)

            # NO CCL — skip steps 8-10

            # Combine: L_AGM = L_CE + λ1 * L_mask
            optimizer.zero_grad()
            L_AGM = L_CE + LAMBDA1 * L_mask
            L_AGM.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss   += L_AGM.item()
            total_l_ce   += L_CE.item()
            total_l_mask += L_mask.item()

        avg_loss   = train_loss   / len(train_loader)
        avg_l_ce   = total_l_ce   / len(train_loader)
        avg_l_mask = total_l_mask / len(train_loader)
        val_f1, val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | L_CE: {avg_l_ce:.4f} | L_mask: {avg_l_mask:.4f} | Val F1: {val_f1:.4f}")
        wandb.log({'epoch': epoch+1, 'train_loss': avg_loss, 'l_ce': avg_l_ce,
                   'l_mask': avg_l_mask, 'val_f1': val_f1, 'val_acc': val_acc})

        if val_f1 > best_val_f1:
            best_val_f1, patience_count = val_f1, 0
            torch.save(model.state_dict(), os.path.join(run_ckpt_dir, 'best_model.pt'))
            print(f'New best model saved at val_f1: {best_val_f1:.4f}')
        else:
            patience_count += 1
            print(f'No improvement {patience_count}/{patience}')
            if patience_count >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    wandb.finish()
    return os.path.join(run_ckpt_dir, 'best_model.pt')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    wandb.login(key=os.environ.get('WANDB_API_KEY'))
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    for target in DOMAINS:
        print(f'\nTarget domain: {target}')
        seed_results = []
        for seed in SEEDS:
            print(f'Seed: {seed}')
            ckpt_path = train(target, seed, tokenizer)
            model = AGMModel(num_labels=2).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

            wandb.init(project='AGM-NLP-2', name=f'ablation1_noccl_{target}_seed{seed}_eval',
                       config={'model': 'agm-ablation1-noccl', 'target_domain': target, 'seed': seed})
            results = evaluate_cross_domain(model, target, tokenizer, seed)
            seed_results.append(results)
            wandb.finish()

        wandb.init(project='AGM-NLP-2', name=f'ablation1_noccl_{target}_summary',
                   config={'model': 'agm-ablation1-noccl', 'target_domain': target,
                           'seeds': SEEDS, 'lambda1': LAMBDA1})
        mean_delta = np.mean([r['delta'] for r in seed_results])
        std_delta  = np.std([r['delta']  for r in seed_results])
        print(f"\n--- Ablation 1 (No CCL) Summary for target: {target} ---")
        print(f"  Δ: {mean_delta:.4f}±{std_delta:.4f}")
        wandb.log({f'summary/mean_delta_{target}': mean_delta, f'summary/std_delta_{target}': std_delta})
        wandb.finish()


if __name__ == '__main__':
    main()