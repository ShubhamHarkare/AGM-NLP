# Attribution-Guided Masking for Robust Cross-Domain Sentiment Classification

**Shubham Harkare · Yash Kulkarni · Arvind Suresh Yogesh Babu**  
University of Michigan — SI 630 NLP Project

> Target venue: EMNLP 2025 Findings / BlackboxNLP 2025 Workshop

---

## Overview

Transformer models such as BERT and RoBERTa achieve strong in-domain performance on sentiment classification but degrade systematically when evaluated on out-of-distribution domains. We show that this degradation is driven by over-reliance on domain-specific spurious tokens rather than domain-invariant sentiment markers.

This repository implements a complete research pipeline:

1. **Baseline Study** — BERT, RoBERTa, DANN, and IRM across all domain-transfer pairs
2. **Diagnostic Study** — Attribution Drift Score (ADS) as a cross-domain diagnostic framework
3. **Attribution-Guided Masking (AGM)** — A novel training objective combining gradient-based spuriousness detection with counterfactual contrastive learning

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
│   ├── agm/
│   │   ├── model.py            # AGMModel
│   │   ├── agm_functions.py    # Helper functions
│   │   └── train.py            # AGM training loop
│   └── ads/
│       ├── compute_mean_ig.py
│       └── ads_pipeline.py
│
├── checkpoints/
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

### Generalization Gap (Δ) — All Models
Lower is better. Δ = |F1_source - F1_target|

| Target | BERT | RoBERTa | DANN | IRM | AGM |
|--------|------|---------|------|-----|-----|
| **IMDb** | 0.097 | 0.095 | 0.012 | 0.049 | **0.013** ✅ |
| **Amazon** | 0.096 | 0.099 | 0.021 | 0.055 | **0.029** ✅ |
| **Hotel** | 0.121 | 0.145 | 0.032 | 0.084 | **0.029** ✅ |
| **Sentiment** | 0.237 | 0.275 | 0.269 | 0.224 | 🔄 Running |

### AGM Training Curves (IMDb fold — representative)

```
Epoch   Loss    L_CE    L_mask  L_CCL   Val F1
1       0.392   0.386   0.042   0.023   0.883
2       0.299   0.293   0.040   0.015   0.901
3       0.225   0.220   0.035   0.014   0.905
8       0.022   0.021   0.005   0.005   0.914
```

L_mask dropped 87% and L_CCL dropped 79% across training — model genuinely learning to suppress spurious token reliance.

---

## Experimental Protocol

- **Task:** Binary sentiment classification (positive / negative)
- **Transfer:** Strict zero-shot — no target domain data during training
- **DANN/IRM/AGM:** Leave-one-out — train on 3 domains, evaluate on 4th
- **Seeds:** [42, 43, 44] — results reported as mean±std
- **Hardware:** University HPC (A100/V100)
- **Tracking:** Weights & Biases project `AGM-NLP`

---

## AGM Loss Function

```
L_AGM = L_CE + λ1·L_mask + λ2·L_CCL

L_CE   = CrossEntropyLoss(logits, labels)
L_mask = mean(attribution[spurious_mask]²)
L_CCL  = ||f(x) - f(x')||²
```

Where x' is a counterfactual generated by replacing spurious tokens via RoBERTa MLM, filtered to preserve sentiment polarity.

---

## Hyperparameters

| Parameter | BERT/RoBERTa | DANN | IRM | AGM |
|-----------|-------------|------|-----|-----|
| Batch size | 32 | 32 | 32 | 16 |
| Max length | 256 | 256 | 256 | 256 |
| LR | 2e-5 | 2e-5 | 2e-5 | 2e-5 |
| Epochs | 10 | 10 | 10 | 10 |
| λ (domain) | — | annealed 0→1 | — | — |
| λ (IRM) | — | — | warmup=1.0, main=1e2 | — |
| λ1 (mask) | — | — | — | 0.1 |
| λ2 (CCL) | — | — | — | 0.1 |

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
python models/bert/train.py
python models/roberta/train.py
python models/dann/train.py
python models/irm/train.py
python models/agm/train.py
python models/ads/ads_pipeline.py
```

---

## Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Setup | ✅ Complete |
| 2 | Baselines (BERT, RoBERTa, DANN, IRM) | ✅ Complete |
| 3 | ADS Diagnostic Study | ✅ Complete (negative result) |
| 4 | AGM Implementation | 🔄 In Progress (3/4 folds) |
| 5 | Ablations | ⏳ Pending |
| 6 | Paper Writing & Submission | 🔄 In Progress |

---

## Notes for Teammates

- **Do not push checkpoints** — large files, gitignored
- **Always run 3 seeds** — single-seed results not accepted
- **AGM batch size is 16** — not 32, due to double backward memory requirement
- **IRM lambda must be 1e2** — 1e4 causes model collapse
- **Log everything to W&B from run 1**