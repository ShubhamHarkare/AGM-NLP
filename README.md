# DRIFT 🌊
### Attribution-Guided Masking for Robust Cross-Domain Sentiment Classification

> SI630 Course Project — University of Michigan
> Shubham Harkare · Yash Kulkarni · Arvind Suresh Yogesh Babu

---

## Overview

Transformer-based models like BERT and RoBERTa achieve strong in-domain performance on sentiment classification but degrade systematically when evaluated on out-of-distribution domains. We hypothesize this is driven by over-reliance on domain-specific spurious tokens rather than domain-invariant sentiment markers.

This project proposes a three-stage framework:
1. **Diagnostic study** — quantify the generalization gap (Δ) and Attribution Drift Score (ADS) across domain-transfer pairs
2. **Predictive experiment** — validate that ADS correlates with Δ prior to any target-domain evaluation
3. **Attribution-Guided Masking (AGM)** — a novel training objective combining Integrated Gradients-based spuriousness detection with a counterfactual contrastive loss to enforce domain-invariant representations

---

## Datasets

| Domain | Dataset | Source | Size (per split) |
|---|---|---|---|
| Movies | `stanfordnlp/imdb` | IMDb reviews | 15,500 |
| Products | `fancyzhx/amazon_polarity` | Amazon reviews | 15,500 |
| Hotels | `enelpol/booking_com_reviews` | Booking.com reviews | 15,500 |
| Social Media | `sentiment140` | Twitter | 15,500 |

Each dataset is standardized to a unified `{text, label, domain}` schema with binary labels `{0, 1}` and split as follows:

| Split | Size | Purpose |
|---|---|---|
| Train | 10,000 | Fine-tune the model |
| Validation | 2,000 | Hyperparameter tuning, early stopping |
| Test | 3,000 | Final F1 evaluation |
| ADS Pool | 500 | Unlabeled target samples for attribution drift computation |

---

## Project Structure

```
DRIFT/
├── raw_data/
│   ├── amazon.py          # Amazon Polarity loader
│   ├── imdb.py            # IMDb loader
│   ├── hotels.py          # Booking.com loader (with preprocessing)
│   ├── sentiment.py       # Sentiment140 loader (with label remapping)
│   └── raw_data_loader.py # Master loader — runs all 4 datasets
├── amazon/                # Saved Amazon splits (train/val/test/ads)
├── hotel/                 # Saved Hotel splits (train/val/test/ads)
├── imdb/                  # Saved IMDb splits (train/val/test/ads)
├── sentiment/             # Saved Sentiment140 splits (train/val/test/ads)
├── validate_datasets.py   # Validation script — checks row counts + schema
├── .env.example           # Environment variable template
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/DRIFT.git
cd DRIFT
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env if needed — defaults are already set for all 4 datasets
```

### 5. Download and prepare all datasets
```bash
python raw_data/raw_data_loader.py
```

### 6. Validate the data
```bash
python validate_datasets.py
```

Expected output:
```
--- Starting Validation ---
Target splits:  10000 Train | 2000 Val | 3000 Test | 500 ADS
Target columns: {'text', 'label', 'domain'}
Target labels:  {0, 1}

✅ amazon: train=10000, val=2000, test=3000, ads=500 | columns=OK | labels=OK
✅ hotel:  train=10000, val=2000, test=3000, ads=500 | columns=OK | labels=OK
✅ imdb:   train=10000, val=2000, test=3000, ads=500 | columns=OK | labels=OK
✅ sentiment: train=10000, val=2000, test=3000, ads=500 | columns=OK | labels=OK

🎉 All datasets validated successfully and match your targets perfectly!
```

---

## Methodology

### Attribution Drift Score (ADS)
For each domain pair (S, T), mean Integrated Gradients (IG) attribution vectors are computed over source and target samples:

```
ADS(S, T) = 1 − cos(meanIG_S, meanIG_T)
```

### Spurious Token Detection
A token `t` is flagged as spurious if:
```
IG_S(t) > τ_high  AND  IG_T(t) < τ_low
```

### AGM Objective
```
L_AGM = L_CE + λ1·L_mask + λ2·L_CCL
```
- **L_CE** — standard cross-entropy on source domain
- **L_mask** — penalizes high attribution on spurious tokens
- **L_CCL** — counterfactual contrastive loss enforcing invariance between original and masked inputs

---

## Baselines

| Model | Description |
|---|---|
| BERT-base | Standard fine-tuned BERT |
| RoBERTa-base | Standard fine-tuned RoBERTa |
| DANN-RoBERTa | Domain-adversarial training |
| IRM-RoBERTa | Invariant Risk Minimization |
| **AGM-RoBERTa** | **Ours — full AGM objective** |

---

## Evaluation Metrics

- **Macro F1** — primary metric
- **Generalization Gap** Δ = |F1_source − F1_target|
- **Transfer Efficiency** TE = F1_target / F1_source
- **Spuriousness Concentration (SC)** — fraction of attribution mass on spurious tokens
- **ADS–Δ Correlation** — Pearson and Spearman correlation across domain pairs

---

## Progress

- [x] Phase 1 — Environment & Data Setup
- [ ] Phase 2 — Baseline Implementation
- [ ] Phase 3 — Diagnostic Study (ADS)
- [ ] Phase 4 — AGM Implementation
- [ ] Phase 5 — Ablations & Analysis
- [ ] Phase 6 — Paper Writing & Submission

---
