# Attribution-Guided Masking for Robust Cross-Domain Sentiment Classification

**Shubham Harkare · Yash Kulkarni · Arvind Suresh Yogesh Babu**  
University of Michigan — SI 630 NLP Project

> Target venue: BlackboxNLP 2026 / EMNLP 2026 (via ARR)

---

## Overview

Transformer models such as BERT and RoBERTa achieve strong in-domain performance on sentiment classification but degrade systematically when evaluated on out-of-distribution domains. We show that this degradation is driven by over-reliance on domain-specific spurious tokens rather than domain-invariant sentiment markers.

This repository implements a complete research pipeline:

1. **Baseline Study** — BERT, RoBERTa, DANN, IRM, Group DRO, and Fish across all domain-transfer pairs
2. **Diagnostic Study** — Attribution Drift Score (ADS) as a cross-domain diagnostic (negative result)
3. **Attribution-Guided Masking (AGM)** — A training-time intervention using gradient-based attribution to detect and penalize spurious token reliance
4. **Ablation Study** — Isolating the contribution of attribution masking vs. contrastive loss

---

## AGM System Architecture

[![](https://mermaid.ink/img/pako:eNpVU9uO2jAQ_ZWRpX1jEQu5EamVIFz2kkjVwkubrJBJJsQicVLbaemy_HudBHYhD1Zmzpkzc0b2kcRlgsQlaV7-jTMqFKxnEY846G8SPvGq1hk8KDi8wf39d5iGr-V0_rqmMOdNqXi7sKct7oWev4I_GKuygb6AWZhTqSBjSYIcpKIKP0vv7mCFXLFCH7AVlMdZB3ht6Tz8Qj2tIlnK8KI-bymL0N9482vBiVKCbWvFSn4jOWv5y3ApaMIaxQN0Lq8KztLLlvoYrqpasLKWsC73evYZKm2v0VW0hm9gmyqDCkXcDJnfuAqo3ENeStllHlvBJz1r0QA_kNOcvSPISwPVNJDXCl5Zc4UipbGqaX7jpFN7DgM_gB1yFHqlEg6bSuhVnS08t5yXY063mGur-FurSDiHHfXUUV8a6sdPlB_gN9v0fJCaLTCBhKUpCuTxjbkF0-NfuVu0vYKwy_s6D_5msgzOozx1cBf4XUB6ZCdYQlwlauyRAkVBm5AcG1pEVIYFRsTVvwkV-4hE_KRrKsp_lWVxKRNlvcuIm2pnOqqrRC9ixuhO0OIzq6fXl7XdJnFNp9Ug7pEciDs0Hvqm6RgDazAcO45tGD3yj7gju2_Z1sgwLVvnH6xTj7y3TQd9Z2xZpjE0nOHItgbjcY9gwvSFD7q31D6p038HkAnm?type=png)](https://mermaid.ai/live/edit#pako:eNpVU9uO2jAQ_ZWRpX1jEQu5EamVIFz2kkjVwkubrJBJJsQicVLbaemy_HudBHYhD1Zmzpkzc0b2kcRlgsQlaV7-jTMqFKxnEY846G8SPvGq1hk8KDi8wf39d5iGr-V0_rqmMOdNqXi7sKct7oWev4I_GKuygb6AWZhTqSBjSYIcpKIKP0vv7mCFXLFCH7AVlMdZB3ht6Tz8Qj2tIlnK8KI-bymL0N9482vBiVKCbWvFSn4jOWv5y3ApaMIaxQN0Lq8KztLLlvoYrqpasLKWsC73evYZKm2v0VW0hm9gmyqDCkXcDJnfuAqo3ENeStllHlvBJz1r0QA_kNOcvSPISwPVNJDXCl5Zc4UipbGqaX7jpFN7DgM_gB1yFHqlEg6bSuhVnS08t5yXY063mGur-FurSDiHHfXUUV8a6sdPlB_gN9v0fJCaLTCBhKUpCuTxjbkF0-NfuVu0vYKwy_s6D_5msgzOozx1cBf4XUB6ZCdYQlwlauyRAkVBm5AcG1pEVIYFRsTVvwkV-4hE_KRrKsp_lWVxKRNlvcuIm2pnOqqrRC9ixuhO0OIzq6fXl7XdJnFNp9Ug7pEciDs0Hvqm6RgDazAcO45tGD3yj7gju2_Z1sgwLVvnH6xTj7y3TQd9Z2xZpjE0nOHItgbjcY9gwvSFD7q31D6p038HkAnm)

## Repository Structure

```
.
├── data/
│   ├── loader.py               # Shared data loading utilities
│   ├── validate_data.py        # Dataset validation
│   └── raw_data/
│       ├── imdb.py
│       ├── amazon.py
│       ├── tripadvisor.py
│       └── sentiment.py
│
├── models/
│   ├── dataset.py              # SentimentDataset — shared across all models
│   ├── bert/
│   │   ├── bert_model.py       # BertClassifier
│   │   └── train.py
│   ├── roberta/
│   │   ├── roberta_model.py    # RobertaClassifier
│   │   └── train.py
│   ├── dann/
│   │   ├── GradientReversalFunction.py
│   │   ├── GradientReversalLayer.py
│   │   ├── DANNModel.py
│   │   └── train.py
│   ├── irm/
│   │   ├── penalty.py          # compute_irm_penalty
│   │   └── train.py
│   ├── dro/
│   │   └── train.py            # Group DRO (Sagawa et al., 2020)
│   ├── fish/
│   │   └── train.py            # Fish gradient matching (Shi et al., 2022)
│   ├── agm/
│   │   ├── model.py            # AGMModel
│   │   ├── agm_functions.py    # Helper functions
│   │   └── train.py            # AGM training loop (Full objective)
│   ├── ablation/
│   │   ├── mask_only.py        # L_CE + λ1·L_mask (no L_CCL)
│   │   ├── no_mask.py          # L_CE + λ2·L_CCL (random token selection)
│   │   └── random_mask.py      # Full objective with random tokens
│   └── ads/
│       ├── compute_mean_ig.py
│       └── ads_pipeline.py
│
├── analysis/
│   └── qualitative_tokens.py   # Token attribution heatmaps (planned)
│
├── checkpoints/                # Selective saves only — see notes
├── results/
├── requirements.txt
└── README.md
```

---

## Datasets

| Domain | Dataset | Style | Notes |
|--------|---------|-------|-------|
| Movie | IMDb Movie Reviews | Long-form narrative | Cinematic vocabulary |
| Product | Amazon Product Reviews | Consumer reviews | Product-specific language |
| Hotel | TripAdvisor Hotel Reviews | Hospitality reviews | Replaces Yelp |
| Social | Sentiment140 (Twitter) | Short informal text | Noisiest domain |

**Split sizes per domain:**

| Split | Size | Notes |
|-------|------|-------|
| Train | 10,000 | Fine-tuning |
| Val | 2,000 | Early stopping |
| Test | 3,000 | Final evaluation |
| ADS | 500 | Carved out first — zero overlap |

---

## Results

### Generalization Gap (Δ) — Baselines (8 seeds)
Lower is better. Δ = |F1_source − F1_target|. All values mean ± std over 8 seeds.

**BERT (single-source, averaged across sources):**

| Target | Avg Δ |
|--------|-------|
| IMDb | 0.119 |
| Amazon | 0.055 |
| Hotel | 0.059 |
| Sent140 | 0.240 |

**RoBERTa (single-source, averaged across sources):**

| Target | Avg Δ |
|--------|-------|
| IMDb | 0.100 |
| Amazon | 0.045 |
| Hotel | 0.060 |
| Sent140 | 0.271 |

**DANN (leave-one-out, 8 seeds):**

| Target | Δ |
|--------|------|
| IMDb | 0.0179±0.0078 |
| Amazon | 0.0246±0.0036 |
| Hotel | 0.0211±0.0118 |
| Sent140 | 0.2641±0.0359 |

**IRM, DRO, Fish** — 🔄 Running (8 seeds)

### AGM Results (3 seeds — to be updated to 8 seeds)

| Target | Full AGM | Mask-only | No Mask | Random |
|--------|----------|-----------|---------|--------|
| IMDb | .011±.005 | .015±.005 | .011±.003 | .010±.002 |
| Amazon | .020±.004 | .020±.004 | .017±.002 | .018±.004 |
| Hotel | .035±.010 | .016±.007 | .030±.011 | .035±.004 |
| Sent140 | .237±.014 | **.232±.010** | .290±.066 | .260±.018 |

### Key Findings (so far)

1. **RoBERTa exploits spurious features more than BERT** — Higher in-domain F1 but worse Sent140 transfer (Δ=0.271 vs 0.240), confirmed with 8 seeds
2. **DANN fails on Sent140** — Strong on structured domains (Δ<0.03) but Δ=0.264 on Twitter transfer
3. **Mask-only AGM is the strongest configuration** — Δ=0.232 on Sent140 with tightest variance (±0.010)
4. **Attribution-guided masking is the critical component** — Removing it (No Mask) degrades by ~6 Δ points; random tokens can't replicate it

---

## AGM Loss Function

```
L_AGM = L_CE + λ1·L_mask + λ2·L_CCL

L_CE   = CrossEntropyLoss(logits, labels)
L_mask = mean(attribution[spurious_mask]²)
L_CCL  = ||f(x) - f(x')||²
```

Where x' is a counterfactual generated by replacing spurious tokens via RoBERTa MLM, filtered to preserve sentiment polarity.

**Main method (Mask-only):** `L = L_CE + λ1·L_mask` (λ2 = 0)

---

## Hyperparameters

| Parameter | BERT/RoBERTa | DANN | IRM | DRO | Fish | AGM |
|-----------|-------------|------|-----|-----|------|-----|
| Batch size | 32 | 32 | 32 | 32 | 32 | 4 (×2 accum = 8 effective) |
| Max length | 256 | 256 | 256 | 256 | 256 | 256 |
| LR | 2e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5 (outer) | 2e-5 |
| Epochs | 50 | 50 | 10 | 50 | 50 | 10 |
| Weight decay | 0.01 | — | 0.01 | 0.01 | 0.01 | 0.01 |
| λ (domain) | — | annealed 0→1 | — | — | — | — |
| λ (IRM) | — | — | warmup=1.0, main=1e2 | — | — | — |
| η (DRO) | — | — | — | 0.01 | — | — |
| C (DRO adj) | — | — | — | 1.5 | — | — |
| Inner LR (Fish) | — | — | — | — | 1e-4 | — |
| λ1 (mask) | — | — | — | — | — | 0.1 |
| λ2 (CCL) | — | — | — | — | — | 0.1 |
| τ_high | — | — | — | — | — | 0.75 |

---

## Experimental Protocol

- **Task:** Binary sentiment classification (positive / negative)
- **Transfer:** Strict zero-shot — no target domain data during training
- **DANN/IRM/DRO/Fish/AGM:** Leave-one-out — train on 3 domains, evaluate on 4th
- **BERT/RoBERTa:** Single-source — train on 1 domain, evaluate on all 4
- **Seeds:** [42, 43, 44, 45, 46, 47, 48, 49] — 8 seeds, reported as mean ± std
- **Statistical tests:** Bootstrap CIs and Wilcoxon signed-rank tests for key comparisons
- **Hardware:** University of Michigan Great Lakes HPC (A100 / V100 / RTX 6000)
- **Tracking:** Weights & Biases project `AGM-NLP-Research`

---

## Checkpoint Strategy

To conserve storage on Great Lakes HPC, we use selective checkpoint saving:

- **BERT, DANN, IRM, DRO, Fish:** No checkpoints saved — best model weights kept in memory during training, metrics logged to wandb
- **RoBERTa:** One checkpoint saved (`imdb`, seed 42) for qualitative token analysis
- **Mask-only AGM:** One checkpoint saved (`sentiment` target fold, seed 42) for qualitative token analysis
- **Full AGM, No Mask, Random Mask:** No checkpoints saved

This saves ~60GB of storage compared to saving all checkpoints.

---

## Setup

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt
cp .env.example .env
# Add WANDB_API_KEY to .env

python data/raw_data/imdb.py
python data/raw_data/amazon.py
python data/raw_data/tripadvisor.py
python data/raw_data/sentiment.py
```

## Training

```bash
# Baselines
python models/bert/train.py
python models/roberta/train.py
python models/dann/train.py
python models/irm/train.py
python models/dro/train.py
python models/fish/train.py

# AGM + Ablations
python models/agm/train.py
python models/ablation/mask_only.py
python models/ablation/no_mask.py
python models/ablation/random_mask.py

# Diagnostics
python models/ads/ads_pipeline.py
```

---

## Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Setup | ✅ Complete |
| 2 | Baselines — BERT, RoBERTa (8 seeds) | ✅ Complete |
| 3 | Baselines — DANN (8 seeds) | ✅ Complete |
| 4 | Baselines — IRM (8 seeds) | 🔄 Running |
| 5 | Baselines — DRO (8 seeds) | 🔄 Running |
| 6 | Baselines — Fish (8 seeds) | 🔄 Running |
| 7 | ADS Diagnostic Study | ✅ Complete (negative result) |
| 8 | AGM + Ablations (8 seeds) | ⏳ Pending |
| 9 | Qualitative Token Analysis | ⏳ Pending |
| 10 | Statistical Tests (bootstrap CIs, Wilcoxon) | ⏳ Pending |
| 11 | Paper Rewrite — Mask-only as main method | ⏳ Pending |
| 12 | Submission to BlackboxNLP 2026 / ARR | ⏳ Target: Aug 2026 |

---

## Notes for Teammates

- **Do not push checkpoints** — large files, gitignored
- **Always run 8 seeds** — [42, 43, 44, 45, 46, 47, 48, 49]
- **AGM batch size is 4 (effective 8)** — due to double backward memory requirement
- **IRM lambda must be 1e2** — 1e4 causes model collapse
- **DRO needs worst-case early stopping** — use min(val_f1) across domains, not average
- **Fish uses Reptile-style inner/outer loop** — not simple gradient averaging
- **Baselines save no checkpoints** — metrics only, best weights in memory
- **Log everything to W&B from run 1**

---

## References

- Sagawa et al. (2020). *Distributionally Robust Neural Networks for Group Shifts.* ICLR.
- Shi et al. (2022). *Gradient Matching for Domain Generalization.* ICLR.
- Ganin et al. (2016). *Domain-Adversarial Training of Neural Networks.* JMLR.
- Arjovsky et al. (2019). *Invariant Risk Minimization.* arXiv.
- Sundararajan et al. (2017). *Axiomatic Attribution for Deep Networks.* ICML.
- Ross et al. (2017). *Right for the Right Reasons.* IJCAI.