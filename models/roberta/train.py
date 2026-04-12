#TODO: Make sure to add the training loop to train the roberta model
#! Make sure to write the loop same as that for models/bert/train.py
import os
import wandb
import sys
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer,get_linear_schedule_with_warmup
from torch.optim import AdamW
from dotenv import load_dotenv
from sklearn.metrics import f1_score,accuracy_score
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.dataset import SentimentDataset
from models.roberta.roberta_model import RobertaClassifier
import numpy as np

# from trconfig import DOMAINS, DATA_ROOT, BATCH_SIZE, MAX_LENGTH, DEVICE
load_dotenv()




import torch
import os


DOMAINS    = ['imdb', 'amazon', 'hotel', 'sentiment']
DATA_ROOT  = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints')
BATCH_SIZE = 32
MAX_LENGTH = 256
EPOCHS     = 50
LR         = 2e-5
WARMUP_RATIO = 0.1
SEEDS      = [42, 43, 44, 45, 46, 47, 48, 49]
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Selective checkpoint config ──────────────────────────────────────
# Only save a checkpoint for this specific (source_domain, seed) combo.
# This one checkpoint is needed later for the qualitative token analysis
# (before-AGM vs after-AGM attribution heatmaps on Sent140 examples).
# Set to None to disable all checkpoint saving.
SAVE_CHECKPOINT_FOR = ('imdb', 42)   # (source_domain, seed)
# ─────────────────────────────────────────────────────────────────────

def get_dataloader(domain, split, tokenizer, shuffle = False):
    path = os.path.join(DATA_ROOT,domain,split)
    dataset = SentimentDataset(path,tokenizer,max_length=MAX_LENGTH)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle = shuffle,
        num_workers=4,
        pin_memory=True # Helps to speed up the CPU->GPU transfer
    )

def evaluate(model,dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            token_type_ids = batch.get('token_type_ids',None)

            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)

            logits = model(input_ids,attention_mask,token_type_ids)
            preds = torch.argmax(logits,dim = 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels,all_preds,average='macro')
    acc = accuracy_score(all_labels,all_preds)

    return f1,acc



def train(source_domain, seed, tokenizer, save_checkpoint=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    run = wandb.init(
        project = 'AGM-NLP-Research',
        name = f'roberta_{source_domain}_seed{seed}',
        config = {
            'model': 'roberta-base',
            'source_domain': source_domain,
            'seed': seed,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'max_length': MAX_LENGTH,
            'epochs': EPOCHS
        }
    )
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    train_loader = get_dataloader(source_domain,'train',tokenizer, shuffle = True)
    for batch in train_loader:
        unique_labels = batch['label'].unique()
        print(f"Unique labels in batch: {unique_labels}")
        break
    val_dataloader = get_dataloader(source_domain,'val',tokenizer,shuffle = False)


    model = RobertaClassifier(num_labels=2).to(DEVICE)

    optimizer = AdamW(model.parameters(),lr = LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()


    # Checkpoint dir — only created if save_checkpoint is True
    run_ckpt_dir = os.path.join(CKPT_DIR, f'roberta_{source_domain}_seed{seed}')
    if save_checkpoint:
        os.makedirs(run_ckpt_dir, exist_ok=True)
        print(f"[CHECKPOINT] Will save best model to {run_ckpt_dir}")

    # early stopping setup
    best_val_f1    = 0.0
    best_state     = None
    patience       = 3
    patience_count = 0


    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch in train_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)
            token_type_ids = batch.get('token_type_ids', None)

            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids)
            loss   = criterion(logits, labels)
            loss.backward()

            # gradient clipping — prevents exploding gradients with RoBERTa
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        val_f1, val_acc = evaluate(model,val_dataloader)


        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        wandb.log({
            'epoch':          epoch + 1,
            'train_loss':     avg_train_loss,
            'val_f1':         val_f1,
            'val_acc':        val_acc,
        })


        # This code below is for early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_count  = 0
            # Save best weights in memory — no disk I/O
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # If this is the run we want to keep, also save to disk
            if save_checkpoint:
                torch.save(
                    model.state_dict(),
                    os.path.join(run_ckpt_dir,'best_model.pt')
                )
                print(f'[CHECKPOINT] New best model saved at: val_f1 of {best_val_f1}')
        else:
            patience_count += 1
            print(f"No improvement {patience_count / patience}")

            if patience_count >= patience:
                print(f'Early stopping triggered at epochs : {epoch + 1}')
                break
    


    wandb.finish()

    # Restore best weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    return model


def evaluate_cross_domain(model, source_domain, tokenizer, seed):
    # seed parameter added so wandb logs are tagged per seed
    model.eval()

    results = {}
    source_f1 = None
    
    for target_domain in DOMAINS:
        test_loader = get_dataloader(
            target_domain, 'test', tokenizer, shuffle=False
        )
        test_f1, test_acc = evaluate(model, test_loader)
        results[target_domain] = {'f1': test_f1, 'acc': test_acc}

        print(f"  {source_domain} → {target_domain}: "
              f"F1={test_f1:.4f}, Acc={test_acc:.4f}")

        # log per-domain test results to wandb
        wandb.log({
            f'test/f1_{source_domain}_to_{target_domain}':  test_f1,
            f'test/acc_{source_domain}_to_{target_domain}': test_acc,
            'seed': seed
        })

        # store source domain F1 for delta computation
        if target_domain == source_domain:
            source_f1 = test_f1

    # compute and log generalization gap (delta) for each transfer pair
    if source_f1 is None:
        raise ValueError(f"source_domain '{source_domain}' not found in DOMAINS")

    for target_domain, metrics in results.items():
        if target_domain != source_domain:
            delta = abs(source_f1 - metrics['f1'])
            te    = metrics['f1'] / source_f1 if source_f1 > 0 else 0.0
            results[target_domain]['delta'] = delta
            results[target_domain]['te']    = te

            # log delta and transfer efficiency — these are your paper's
            # core diagnostic metrics so they need to be logged clearly
            wandb.log({
                f'delta/{source_domain}_to_{target_domain}': delta,
                f'te/{source_domain}_to_{target_domain}':    te,
                'seed': seed
            })

            print(f"  Δ ({source_domain}→{target_domain}): "
                  f"{delta:.4f} | TE: {te:.4f}")

    return results


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    wandb.login(key=os.environ.get('WANDB_API_KEY'))


    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    for source_domain in DOMAINS:
        print(f'Training on source domain: {source_domain}')

        seed_results = []
        #DESC: Making sure that we are not lucky because of the seed, hence we are repeating our experiment so that we can have a error margin
        for seed in SEEDS:
            print(f'SEED: {seed}')

            # Only save checkpoint for the specific combo we need for qualitative analysis
            should_save = (
                SAVE_CHECKPOINT_FOR is not None
                and source_domain == SAVE_CHECKPOINT_FOR[0]
                and seed == SAVE_CHECKPOINT_FOR[1]
            )

            model = train(
                source_domain=source_domain,
                seed=seed,
                tokenizer=tokenizer,
                save_checkpoint=should_save
            )

            #DESC: The below code is to log the results onto weights and biases 
            wandb.init(
                project = 'AGM-NLP-Research',
                name=f'roberta_{source_domain}_seed{seed}_eval',
                config= {
                    'model': 'roberta-base',
                    'source_domain': source_domain,
                    'seed': seed
                }
            )

            results = evaluate_cross_domain(model,source_domain,tokenizer,seed)
            seed_results.append(results)
            wandb.finish()

            # Explicitly free GPU memory before next seed
            del model
            torch.cuda.empty_cache()


            
        wandb.init(
            project='AGM-NLP-Research',
            name= f'roberta_{source_domain}_summary',
            config  = {
            'model':         'roberta-base',
            'source_domain': source_domain,
            'seeds':         SEEDS,
            'batch_size':    BATCH_SIZE,
            'lr':            LR,
            'max_length':    MAX_LENGTH,
            'epochs':        EPOCHS,
            }
        )

        print(f"\n--- Summary across seeds for {source_domain} ---")

        for target_domain in DOMAINS:
            f1_scores  = [r[target_domain]['f1']  for r in seed_results]
            acc_scores = [r[target_domain]['acc'] for r in seed_results]

            mean_f1  = np.mean(f1_scores)
            std_f1   = np.std(f1_scores)
            mean_acc = np.mean(acc_scores)
            std_acc  = np.std(acc_scores)

            print(f"  {source_domain} → {target_domain}: "
                f"F1={mean_f1:.4f}±{std_f1:.4f} | "
                f"Acc={mean_acc:.4f}±{std_acc:.4f}")

            wandb.log({
                f'summary/mean_f1_{source_domain}_to_{target_domain}':  mean_f1,
                f'summary/std_f1_{source_domain}_to_{target_domain}':   std_f1,
                f'summary/mean_acc_{source_domain}_to_{target_domain}': mean_acc,
                f'summary/std_acc_{source_domain}_to_{target_domain}':  std_acc,
            })

            if target_domain != source_domain:
                delta_scores = [r[target_domain]['delta'] for r in seed_results]
                te_scores    = [r[target_domain]['te']    for r in seed_results]

                mean_delta = np.mean(delta_scores)
                std_delta  = np.std(delta_scores)
                mean_te    = np.mean(te_scores)
                std_te     = np.std(te_scores)

                print(f"  Δ ({source_domain}→{target_domain}): "
                    f"{mean_delta:.4f}±{std_delta:.4f} | "
                    f"TE: {mean_te:.4f}±{std_te:.4f}")

                wandb.log({
                    f'summary/mean_delta_{source_domain}_to_{target_domain}': mean_delta,
                    f'summary/std_delta_{source_domain}_to_{target_domain}':  std_delta,
                    f'summary/mean_te_{source_domain}_to_{target_domain}':    mean_te,
                    f'summary/std_te_{source_domain}_to_{target_domain}':     std_te,
                })

        wandb.finish() 
        print(f"\n Completed all seeds for source domain: {source_domain}")





if __name__ == '__main__':
    main()