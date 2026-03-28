# DESC: This file is responsible for computing the Attribution Drift Score (ADS)
# for all 12 transfer pairs and correlating it with the generalization gap (Δ)
# This is Phase 3 of the AGM project — the diagnostic study

import os
import sys
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.roberta.roberta_model import RobertaClassifier
from models.dataset import SentimentDataset
# from analysis.ig_computation import compute_mean_ig, compute_ads
from models.ads.compute_mean_ig import compute_token_ig, compute_ads_shared_vocab
# ── Config ────────────────────────────────────────────────────────────────────

DOMAINS    = ['imdb', 'amazon', 'hotel', 'sentiment']
DATA_ROOT  = os.path.join(os.path.dirname(__file__), '..','..', 'data')
CKPT_DIR   = os.path.join(os.path.dirname(__file__), '..','..', 'checkpoints')
MAX_LENGTH = 256
BATCH_SIZE = 4   # smaller batch size for IG computation — memory intensive
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED       = 42   # use seed 42 checkpoint for ADS computation

# ── RoBERTa Δ values from Phase 2 results ─────────────────────────────────────
# These are mean Δ values across 3 seeds from your RoBERTa baseline

ROBERTA_DELTA = {
    ('imdb',      'amazon'):    0.006,
    ('imdb',      'hotel'):     0.105,
    ('imdb',      'sentiment'): 0.298,
    ('amazon',    'imdb'):      0.047,
    ('amazon',    'hotel'):     0.046,
    ('amazon',    'sentiment'): 0.245,
    ('hotel',     'imdb'):      0.235,
    ('hotel',     'amazon'):    0.104,
    ('hotel',     'sentiment'): 0.283,
    ('sentiment', 'imdb'):      0.023,
    ('sentiment', 'amazon'):    0.047,
    ('sentiment', 'hotel'):     0.023,
}


#DESC: The below is the function to load the ads data
def get_ads_dataloader(domain,tokenizer):
    '''
    Loads the ADS split (500 samples) for a given domain.
    No domain ID needed — ADS computation only needs text.
    '''
    path = os.path.join(DATA_ROOT,domain,'ads')
    dataset = SentimentDataset(path,tokenizer,max_length=MAX_LENGTH)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle = False,
        num_workers=4,
        pin_memory=True
    )


def load_roberta(domain,seed = SEED):
    '''
    Loads the trained RoBERTa checkpoint for a given source domain.
    Uses seed 42 by default for ADS computation.
    '''

    ckpt_path = os.path.join(
        CKPT_DIR,
        f'roberta_{domain}_seed{seed}',
        'best_model.pt'
    )

    model = RobertaClassifier(num_labels=2).to(DEVICE)
    model.load_state_dict(
        torch.load(ckpt_path,map_location=DEVICE)
    )

    model.eval()
    print(f'Loaded RoBERTa checkpoint for domain: {domain}')
    return model


def main():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    print("\n=== Computing directional token-level ADS ===")
    ads_scores = {}
    shared_token_counts = {}

    for source in DOMAINS:
        print(f"\nLoading source model: {source}")
        model = load_roberta(source)

        # compute token IG on source domain data using source model
        print(f"  Computing IG on {source} data...")
        source_loader = get_ads_dataloader(source, tokenizer)
        token_ig_source = compute_token_ig(model, source_loader, DEVICE)

        # compute token IG on each target domain using same source model
        for target in DOMAINS:
            if source == target:
                continue
            print(f"  Computing IG on {target} data (source model: {source})...")
            target_loader = get_ads_dataloader(target, tokenizer)
            token_ig_target = compute_token_ig(model, target_loader, DEVICE)

            # compute ADS using shared vocabulary only
            ads, n_shared = compute_ads_shared_vocab(token_ig_source, token_ig_target)
            ads_scores[(source, target)] = ads
            shared_token_counts[(source, target)] = n_shared
            print(f"  ADS ({source}→{target}): {ads:.4f} | shared tokens: {n_shared}")

        del model
        torch.cuda.empty_cache()

    # build parallel lists for correlation
    print("\n=== Correlating ADS with Δ ===")
    ads_list    = []
    delta_list  = []
    pair_labels = []

    for pair, ads in ads_scores.items():
        if pair in ROBERTA_DELTA:
            ads_list.append(ads)
            delta_list.append(ROBERTA_DELTA[pair])
            pair_labels.append(f"{pair[0]}→{pair[1]}")

    ads_array   = np.array(ads_list)
    delta_array = np.array(delta_list)

    pearson_r,  pearson_p  = pearsonr(ads_array, delta_array)
    spearman_r, spearman_p = spearmanr(ads_array, delta_array)

    print(f"\nPearson  r = {pearson_r:.4f}  (p = {pearson_p:.4f})")
    print(f"Spearman r = {spearman_r:.4f}  (p = {spearman_p:.4f})")

    # save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, 'ads_results.txt')
    with open(results_path, 'w') as f:
        f.write("ADS Results — Phase 3 Diagnostic Study (Shared Vocabulary)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Transfer Pair':<20}  {'ADS':>8}  {'Δ (RoBERTa)':>12}  {'Shared Tokens':>14}\n")
        f.write("-" * 60 + "\n")
        for i, pair in enumerate(pair_labels):
            n_shared = shared_token_counts.get(
                tuple(pair.replace('→', '->').split('->')), 0
            )
            f.write(f"{pair:<20}  {ads_list[i]:>8.4f}  {delta_list[i]:>12.4f}  {n_shared:>14}\n")
        f.write("\n")
        f.write(f"Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.4f})\n")
        f.write(f"Spearman r = {spearman_r:.4f}  (p = {spearman_p:.4f})\n")

    print(f"Results saved to {results_path}")

    # scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(ads_array, delta_array, color='steelblue', s=80, zorder=5)
    for i, label in enumerate(pair_labels):
        plt.annotate(label, (ads_array[i], delta_array[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    m, b = np.polyfit(ads_array, delta_array, 1)
    x_line = np.linspace(ads_array.min(), ads_array.max(), 100)
    plt.plot(x_line, m * x_line + b, color='red', linewidth=1.5, linestyle='--')
    plt.xlabel('Attribution Drift Score (ADS)', fontsize=12)
    plt.ylabel('Generalization Gap (Δ)', fontsize=12)
    plt.title(
        f'ADS vs Generalization Gap (Shared Vocabulary)\n'
        f'Pearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}',
        fontsize=13
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'ads_vs_delta.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Scatter plot saved to {plot_path}")


if __name__ == '__main__':
    main()

