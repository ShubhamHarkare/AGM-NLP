
#DESC: This file contains the training and evaluation loop for the IRM function over the RoBERTa model.

import os
import wandb
import sys
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer,get_linear_schedule_with_warmup
from torch.optim import AdamW
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from dotenv import load_dotenv
from sklearn.metrics import f1_score,accuracy_score

# from models.roberta.roberta_model. import RobertaClassifier
# from models.roberta.roberta_model import RobertaClassifier
from models.roberta.roberta_model import RobertaClassifier
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.irm.penalty import compute_irm_penalty #! Important for calculating the IRM penalty



from models.dataset import SentimentDataset

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
EPOCHS     = 10
LR         = 2e-5
WARMUP_RATIO = 0.1
SEEDS      = [42, 43, 44]
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IRM_LAMBDA_WARMUP = 1.0
IRM_LAMBDA_MAIN = 1e2
WARMUP_STEPS = 500

def get_dataloader(target_domain,split,tokenizer,shuffle = False):
    datasets = []
    for domain in DOMAINS:
        if domain == target_domain: continue

        path = os.path.join(DATA_ROOT,domain,split)
        dataset = SentimentDataset(path,tokenizer,max_length=MAX_LENGTH)

        dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=shuffle,num_workers=4,pin_memory=True)
        datasets.append(dataloader)

    return datasets


def get_target_dataloader(domain, split, tokenizer, shuffle=False):
    path = os.path.join(DATA_ROOT, domain, split)
    dataset = SentimentDataset(path, tokenizer, max_length=MAX_LENGTH)
    # Note: no domain ID needed — evaluate() only uses sentiment labels
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
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

def evaluate_cross_domain(model, target_domain, tokenizer, seed):
    model.eval()

    # Step 1: evaluate on combined training domains test set
    # get_dataloader returns list of 3 DataLoaders — iterate over all 3
    train_test_loaders = get_dataloader(target_domain, 'test', tokenizer, shuffle=False)
    
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for loader in train_test_loaders:
            for batch in loader:
                input_ids      = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels         = batch['label'].to(DEVICE)
                token_type_ids = batch.get('token_type_ids', None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(DEVICE)
                logits = model(input_ids, attention_mask, token_type_ids)
                preds  = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    source_f1  = f1_score(all_labels, all_preds, average='macro')
    source_acc = accuracy_score(all_labels, all_preds)

    # Step 2: evaluate on held-out target domain test set
    target_loader = get_target_dataloader(target_domain, 'test', tokenizer, shuffle=False)
    target_f1, target_acc = evaluate(model, target_loader)

    # Step 3: compute delta and TE
    delta = abs(source_f1 - target_f1)
    te    = target_f1 / source_f1 if source_f1 > 0 else 0.0

    # Step 4: print results
    print(f"  Source (combined 3 domains) F1 : {source_f1:.4f}")
    print(f"  Target ({target_domain}) F1    : {target_f1:.4f}")
    print(f"  Δ: {delta:.4f} | TE: {te:.4f}")

    # Step 5: log to wandb
    wandb.log({
        f'test/source_f1':                  source_f1,
        f'test/target_f1_{target_domain}':  target_f1,
        f'delta/{target_domain}':           delta,
        f'te/{target_domain}':              te,
        'seed': seed
    })

    return {
        'source_f1': source_f1,
        'target_f1': target_f1,
        'delta':     delta,
        'te':        te
    }


def train(domain,seed,tokenizer):
    torch.manual_seed(seed=seed)
    np.random.seed(seed=seed)


    run = wandb.init(
        project='AGM-NLP',
        name = f'IRM_{domain}_seed{seed}',
        config={
            'model': 'irm-roberta',
            'source_domain': domain,
            'seed': seed,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'max_length': MAX_LENGTH,
            'epochs': EPOCHS
        }
    )

    train_loader = get_dataloader(domain,'train',tokenizer,shuffle=True)
    val_dataloader = get_target_dataloader(domain, 'val', tokenizer, shuffle=False)

    model = RobertaClassifier(num_labels=2).to(DEVICE)

    optimizer = AdamW(model.parameters(),lr = LR, weight_decay=0.01)
    total_steps = len(train_loader[0]) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()

    run_ckpt_dir = os.path.join(CKPT_DIR, f'irm_{domain}_seed{seed}')
    os.makedirs(run_ckpt_dir, exist_ok=True)

    # early stopping setup
    best_val_f1    = 0.0
    patience       = 3
    patience_count = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0


        for batch_idx,batches in enumerate(zip(*train_loader)):
            total_erm = 0
            total_penalty = 0


            for batch in batches:
                input_ids      = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels         = batch['label'].to(DEVICE)
                token_type_ids = batch.get('token_type_ids', None)

                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(DEVICE)

                
                logits = model(input_ids,attention_mask,token_type_ids)
                erm_loss = criterion(logits,labels)
                penalty = compute_irm_penalty(logits,labels,criterion)
                total_erm += erm_loss
                total_penalty += penalty

            current_step = epoch * len(train_loader[0]) + batch_idx
            lambda_ = IRM_LAMBDA_WARMUP if current_step < WARMUP_STEPS else IRM_LAMBDA_MAIN
            total_loss = total_erm + lambda_ * total_penalty
            optimizer.zero_grad()
            total_loss.backward()

            optimizer.step()
            scheduler.step()
            train_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader[0])
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
            'lambda': lambda_
        })


        # This code below is for early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_count  = 0
            torch.save(
                model.state_dict(),
                os.path.join(run_ckpt_dir,'best_model.pt')
            )
            print(f'New best model saved at: val_f1 of {best_val_f1}')
        else:
            patience_count += 1
            print(f"No improvement {patience_count / patience}")

            if patience_count >= patience:
                print(f'Early stopping triggered at epochs : {epoch + 1}')
                break
    


    wandb.finish()
    # model.load_state_dict(
    #     torch.load(os.path.join(run_ckpt_dir,'best_model.pt'))
    # )
    return os.path.join(run_ckpt_dir, 'best_model.pt')

        



def main():
    os.makedirs(CKPT_DIR,exist_ok=True)
    wandb.login(key=os.environ.get('WANDB_API_KEY'))


    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    for target in DOMAINS:
        print(f'Training on source domain: {target}')

        seed_results = []
        #DESC: Making sure that we are not lucky because of the seed, hence we are repeating our experiment so that we can have a error margin
        for seed in SEEDS:
            print(f'SEED: {seed}')

            ckpt_path = train(domain=target,seed=seed,tokenizer=tokenizer)
            model = RobertaClassifier(num_labels=2).to(DEVICE)

            model.load_state_dict(
                torch.load(ckpt_path,map_location=DEVICE)
            )
            #DESC: The below code is to log the results onto weights and biases 
            wandb.init(
                project = 'AGM-NLP',
                name=f'irm_{target}_seed{seed}_eval',
                config= {
                    'model': 'irm_roberta',
                    'source_domain': target,
                    'seed': seed
                }
            )

            results = evaluate_cross_domain(model,target,tokenizer,seed)
            seed_results.append(results)
            wandb.finish()



            #DESC: This is the logging for the summary after we have completed everything
        wandb.init(
            project='AGM-NLP',
            name= f'irm_{target}_summary',
            config  = {
            'model':         'irm_roberta',
            'source_domain': target,
            'seeds':         SEEDS,
            'batch_size':    BATCH_SIZE,
            'lr':            LR,
            'max_length':    MAX_LENGTH,
            'epochs':        EPOCHS,
            }
        )

        print(f"\n--- Summary across seeds for target: {target} ---")

        mean_source_f1 = np.mean([r['source_f1'] for r in seed_results])
        std_source_f1  = np.std([r['source_f1']  for r in seed_results])
        mean_target_f1 = np.mean([r['target_f1'] for r in seed_results])
        std_target_f1  = np.std([r['target_f1']  for r in seed_results])
        mean_delta     = np.mean([r['delta']     for r in seed_results])
        std_delta      = np.std([r['delta']      for r in seed_results])
        mean_te        = np.mean([r['te']        for r in seed_results])
        std_te         = np.std([r['te']         for r in seed_results])

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

        wandb.finish()  # close summary run before moving to next source domain
        print(f"\n Completed all seeds for source domain: {target}")





if __name__ == '__main__':
    main()




