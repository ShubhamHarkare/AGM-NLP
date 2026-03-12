# Attribution-Guided Masking for Robust Cross-Domain Sentiment Classification

**Shubham Harkare · Yash Kulkarni · Arvind Suresh Yogesh Babu**  
University of Michigan — SI 630 NLP Project

---

## Overview

Transformer models such as BERT and RoBERTa achieve strong in-domain performance on sentiment classification but degrade systematically when evaluated on out-of-distribution domains. We hypothesize that this degradation is driven by over-reliance on domain-specific spurious tokens rather than domain-invariant sentiment markers.

This repository implements a three-stage framework:

1. **Diagnostic Study** — Quantifying the generalization gap (Δ) and Attribution Drift Score (ADS) across domain-transfer pairs
2. **Predictive Experiment** — Validating that ADS correlates with Δ prior to any target-domain evaluation
3. **Attribution-Guided Masking (AGM)** — A novel training objective combining Integrated Gradients-based spuriousness detection with a counterfactual contrastive loss

Target venue: EMNLP Findings or ACL Findings

---

## Repository Structure

```
.
├── data/
│   ├── loader.py               # Shared data loading utilities
│   ├── validate_data.py        # Dataset validation and sanity checks
│   └── raw_data/               # Domain-specific download scripts
│       ├── imdb.py             # IMDb Movie Reviews
│       ├── amazon.py           # Amazon Product Reviews
│       ├── tripadvisor.py      # TripAdvisor Hotel Reviews
│       └── sentiment.py        # Sentiment140 (Twitter)
│
├── models/
│   ├── dataset.py              # Shared SentimentDataset class (all models)
│   ├── bert/
│   │   ├── bert_model.py       # BertClassifier — BertModel + linear head
│   │   └── train.py            # BERT training loop, evaluation, W&B logging
│   ├── roberta/
│   │   ├── roberta_model.py    # RobertaClassifier — RobertaModel + linear head
│   │   └── train.py            # RoBERTa training loop (mirrors BERT exactly)
│   └── dann/
│       ├── GradientReversalFunction.py   # Custom autograd function (gradient flip)
│       ├── GradientReversalLayer.py      # nn.Module wrapper for GRL
│       ├── DANNModel.py                  # DANN — RoBERTa + sentiment head + domain head
│       └── train.py                      # DANN training loop with dual loss
│
├── checkpoints/                # Saved model weights (per domain, per seed)
├── requirements.txt
└── README.md
```

---

## Datasets

All datasets are standardized to `{text, label, domain}` format with binary labels {0, 1}, truncated to 256 tokens, and split into train/val/test.

| Domain | Dataset | Size | Notes |
|--------|---------|------|-------|
| Movie | IMDb Movie Reviews | Balanced | Long-form narrative reviews |
| Product | Amazon Product Reviews | Balanced | Consumer product reviews |
| Hotel | TripAdvisor Hotel Reviews | Balanced | Replaces Yelp from original proposal |
| Social | Sentiment140 (Twitter) | Balanced | Short informal text, noisiest domain |

Data is stored in HuggingFace Arrow format (`load_from_disk`) under `data/<domain>/train`, `data/<domain>/val`, `data/<domain>/test`.

---

## Models

### Implemented

| Model | File | Status | Notes |
|-------|------|--------|-------|
| BERT | `models/bert/` | ✅ Complete | Baseline, results logged |
| RoBERTa | `models/roberta/` | ✅ Complete | Stronger baseline, results logged |
| DANN-RoBERTa | `models/dann/` | 🔄 Training | Leave-one-out, multi-source adversarial |
| IRM-RoBERTa | `models/irm/` | ⏳ Pending | Invariant Risk Minimization |
| AGM-RoBERTa | `models/agm/` | ⏳ Pending | Main contribution |

### Planned (Phase 3–4)

- `models/agm/` — Attribution-Guided Masking with counterfactual contrastive loss
- `analysis/ads.py` — Attribution Drift Score computation using Captum

---

## Experimental Protocol

**Task:** Binary sentiment classification (positive / negative)

**Transfer protocol:** Strict zero-shot — models are fine-tuned on source domain only and evaluated directly on unseen target domains. No target domain samples are used at any point during training.

**Seeds:** All experiments run with seeds [42, 43, 44]. Results reported as mean ± std.

**Hardware:** University HPC cluster (A100/V100 GPU)

**Key metrics:**
- Macro F1 (primary)
- Accuracy
- Generalization Gap: `Δ = |F1_source - F1_target|`
- Transfer Efficiency: `TE = F1_target / F1_source`

**DANN-specific protocol:** Leave-one-out multi-source training. Each fold trains on 3 domains combined and evaluates zero-shot on the held-out 4th domain. Domain classifier uses `num_domains=3` per fold and is discarded at evaluation time.

---

## Baseline Results

### BERT — Generalization Gap (Δ) Matrix

| Source \ Target | IMDb | Amazon | Hotel | Sentiment |
|----------------|------|--------|-------|-----------|
| **IMDb** | — | 0.008 | 0.046 | 0.226 |
| **Amazon** | 0.054 | — | 0.043 | 0.229 |
| **Hotel** | 0.194 | 0.092 | — | 0.255 |
| **Sentiment** | 0.042 | 0.017 | 0.070 | — |

### RoBERTa — Generalization Gap (Δ) Matrix

| Source \ Target | IMDb | Amazon | Hotel | Sentiment |
|----------------|------|--------|-------|-----------|
| **IMDb** | — | 0.006 | 0.105 | 0.298 |
| **Amazon** | 0.047 | — | 0.046 | 0.245 |
| **Hotel** | 0.235 | 0.104 | — | 0.283 |
| **Sentiment** | 0.023 | 0.047 | 0.023 | — |

**Key finding:** RoBERTa achieves higher in-domain F1 across all domains but shows larger generalization gaps than BERT on 8 of 12 transfer pairs — particularly on stylistically distant domain pairs. This suggests increased model capacity amplifies spurious domain-specific correlations during fine-tuning, strongly motivating AGM.

---

## Training

### BERT / RoBERTa

```bash
python models/bert/train.py
python models/roberta/train.py
```

Loops over all 4 source domains, trains 3 seeds each, logs to W&B.

### DANN

```bash
python models/dann/train.py
```

Leave-one-out protocol — trains 4 folds × 3 seeds = 12 total runs.

---

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd <repo>

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your WANDB_API_KEY to .env

# Download and preprocess datasets
python data/raw_data/imdb.py
python data/raw_data/amazon.py
python data/raw_data/tripadvisor.py
python data/raw_data/sentiment.py
```

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 32 | All models |
| Max length | 256 | Tokens |
| Learning rate | 2e-5 | AdamW |
| Warmup ratio | 0.1 | Linear warmup |
| Epochs | 10 | With early stopping |
| Early stopping patience | 3 | On val F1 |
| Gradient clip | 1.0 | Max norm |
| Seeds | 42, 43, 44 | 3 runs per experiment |
| DANN λ schedule | Annealed 0→1 | `2/(1+exp(-10p))-1` |

---

## Experiment Tracking

All runs logged to Weights & Biases under project `AGM-NLP`. Each run logs:
- Per-epoch train loss, val F1, val accuracy
- Per-transfer-pair test F1, accuracy, Δ, TE
- Summary statistics (mean ± std) across seeds

---

## Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment & Data Setup | ✅ Complete |
| 2 | Baseline Implementation (BERT, RoBERTa, DANN, IRM) | 🔄 In Progress |
| 3 | Diagnostic Study — ADS computation and correlation with Δ | ⏳ Pending |
| 4 | AGM Implementation — masking loss + counterfactual contrastive loss | ⏳ Pending |
| 5 | Ablations & Analysis | ⏳ Pending |
| 6 | Paper Writing & Submission | ⏳ Pending |

---

## Citation

If you reference this work, please cite:

```
Harkare, S., Kulkarni, Y., & Suresh Yogesh Babu, A. (2025).
Attribution-Guided Masking for Robust Cross-Domain Sentiment Classification.
University of Michigan, SI 630.
```

---

## Notes for Teammates

- **Do not push checkpoints to GitHub** — they are large and gitignored
- **Always run 3 seeds** — single-seed results will not be accepted in the paper
- **Log everything to W&B from run 1** — do not rely on terminal output
- **Match hyperparameters exactly across models** — reviewers will check for fair comparison
- **Document any deviation from the protocol** — especially dataset or preprocessing changes