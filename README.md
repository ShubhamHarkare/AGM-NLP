# Attribution-Guided Masking for Robust Cross-Domain Sentiment Classification

**Shubham Harkare · Yash Kulkarni · Arvind Suresh Yogesh Babu**
University of Michigan — SI 630 NLP Project


---

## Overview

Transformer models such as BERT and RoBERTa achieve strong in-domain performance on sentiment classification but degrade systematically when evaluated on out-of-distribution domains. We show this degradation is driven by over-reliance on domain-specific spurious tokens rather than domain-invariant sentiment markers.

This repository implements a complete research pipeline:

1. **Baseline Study** — BERT, RoBERTa, DANN, IRM, Group DRO, and Fish across all domain-transfer pairs
2. **Diagnostic Study** — Attribution Drift Score (ADS) as a cross-domain diagnostic (**negative result** — see below)
3. **Attribution-Guided Masking (AGM)** — A training-time intervention using gradient-based attribution to detect and penalize spurious token reliance
4. **Ablation Study** — Isolating the contribution of attribution-guided masking vs. random-token masking vs. the counterfactual contrastive loss

The paper's central thesis: ADS **fails** as a post-hoc diagnostic for predicting cross-domain generalization failure, but the same attribution signal **succeeds** when used as a training-time regularizer (AGM). This structural contrast — not just the headline numbers — is the core contribution.

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
│   │   └── train.py            # NOTE: early stopping uses source-domain val loader only
│   ├── dro/
│   │   └── train.py            # Group DRO (Sagawa et al., 2020)
│   ├── fish/
│   │   └── train.py            # Fish gradient matching (Shi et al., 2022)
│   ├── agm/
│   │   ├── model.py            # AGMModel
│   │   ├── agm_functions.py    # Helper functions
│   │   └── train.py            # AGM training loop (Full objective)
│   ├── ablation/
│   │   ├── mask_only.py        # L_CE + λ1·L_mask (no L_CCL) — main method
│   │   ├── no_mask.py          # L_CE + λ2·L_CCL (random token selection)
│   │   └── random_mask.py      # Full objective with random tokens
│   └── ads/
│       ├── compute_mean_ig.py
│       └── ads_pipeline.py
│
├── analysis/
│   └── qualitative_tokens.py   # Token attribution heatmaps
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

## Final Results (8 seeds, leave-one-out zero-shot protocol)

### Generalization Gap (Δ = \|F1_source − F1_target\|), lower is better

| Target | BERT | RoBERTa | DANN | IRM | DRO | Fish | AGM (full) | **AGM (mask-only)** |
|--------|------|---------|------|-----|-----|------|------------|----------------------|
| IMDb | .119 | .100 | .018 | .024 | .021 | .017 | .017 | **.013** |
| Amazon | .055 | .045 | .025 | .034 | .033 | .027 | .021 | **.019** |
| Hotel | .059 | .060 | .021 | .017 | .017 | .025 | .031 | .032 |
| Sent140 | .240 | .271 | .264 | .238 | .248 | .247 | .244 | .244 |

BERT/RoBERTa values are averaged across all source–target pairs for each target; all other methods are mean over 8 seeds (42–49). Mask-only AGM (`L = L_CE + λ1·L_mask`) is the recommended default configuration.

### The Sentiment140 story: mean ranking vs. variance

On the hardest transfer (Sentiment140), AGM and IRM are **statistically indistinguishable** in mean Δ (0.244 vs. 0.238, overlapping 95% bootstrap CIs, 10,000 resamples). The differentiator is **variance and stability**:

| Method | Sent140 Δ (mean ± std, 8 seeds) |
|--------|----------------------------------|
| DANN | 0.264 ± 0.036 |
| DRO | 0.248 ± 0.029 |
| Fish | 0.247 ± 0.021 |
| IRM | 0.238 ± 0.036 |
| **AGM mask-only** | **0.244 ± 0.023** |

IRM required extensive hyperparameter search (an initial configuration with λ=10⁴ failed entirely) and carries the highest variance among all methods. AGM mask-only uses a single stable hyperparameter (λ1) across all folds and achieves competitive mean performance with markedly tighter variance — a stability–performance trade-off that matters in practice.

### Ablation: attribution-guided vs. random masking (8 seeds)

| Target | Mask-only (AGM) | Random Mask |
|--------|------------------|-------------|
| IMDb | .013 ± .001 | .016 ± .005 |
| Amazon | .019 ± .004 | .018 ± .004 |
| Hotel | .032 ± .010 | .033 ± .009 |
| Sent140 | .244 ± .023 | .261 ± .020 |

On Sentiment140, attribution-guided token selection reduces Δ by 0.017 absolute versus random selection at the same masking rate — the critical component is *which* tokens get masked, not just that masking occurs.

### Key Findings

1. **RoBERTa exploits spurious features more than BERT** — higher in-domain F1 but worse Sentiment140 transfer (Δ=0.271 vs. 0.240), consistent with capacity amplifying reliance on spurious correlations during fine-tuning.
2. **DANN, DRO, and Fish all degrade sharply on Sentiment140** relative to their strong performance on structured domains (Δ<0.03), while IRM is more uniform but weaker on structured domains.
3. **Mask-only AGM is the strongest configuration overall** — best or near-best Δ on 3 of 4 targets, with the tightest variance of any competitive method on Sentiment140.
4. **Attribution-guided masking is the critical component** — random-token masking cannot replicate the Sentiment140 gains, and removing masking entirely degrades performance further.
5. **Attribution Drift Score (ADS) fails as a diagnostic** — none of three formulations (symmetric, directional, shared-vocabulary) correlates meaningfully with the generalization gap across the 12 transfer pairs (best Pearson r = 0.21, shared-vocab r = 0.04 / Spearman r = −0.05). ADS values cluster tightly in [0.72, 0.96] while Δ spans [0.01, 0.30] — insufficient variance to be predictive. This negative result motivates using the same attribution signal as a *training-time* regularizer instead of a *post-hoc* diagnostic.

---

## AGM Loss Function

```
L_AGM = L_CE + λ1·L_mask + λ2·L_CCL

L_CE   = CrossEntropyLoss(logits, labels)
L_mask = mean(attribution[spurious_mask]²)
L_CCL  = ||f(x) - f(x')||²
```

Where x' is a counterfactual generated by replacing high-attribution tokens via RoBERTa MLM, filtered to preserve the model's predicted sentiment polarity.

**Main method (Mask-only, recommended default):** `L = L_CE + λ1·L_mask` (λ2 = 0)

The counterfactual contrastive loss (L_CCL) is explored as an auxiliary term; ablations show it provides no consistent improvement over mask-only and is retained in the paper as an appendix formulation for completeness.

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
| λ (IRM) | — | — | warmup=500, main=1e2 | — | — | — |
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
- **Statistical tests:** 95% bootstrap CIs (10,000 resamples) for the closest mean comparison (AGM vs. IRM on Sentiment140); differences fall within overlapping intervals and are reported as statistically indistinguishable
- **Hardware:** University of Michigan Great Lakes HPC (A100 / V100 / RTX 6000)
- **Tracking:** Weights & Biases project `AGM-NLP-Research`

### Known issue, fixed prior to submission
An early-stopping bug in the IRM training loop (`get_target_dataloader` was being used for early stopping instead of the source-domain validation loader) was identified and corrected. Re-running with the fix produced essentially unchanged IRM numbers; the results above reflect the corrected pipeline.

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

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Setup | ✅ Complete |
| 2 | Baselines — BERT, RoBERTa (8 seeds) | ✅ Complete |
| 3 | Baselines — DANN, IRM, DRO, Fish (8 seeds) | ✅ Complete |
| 4 | ADS Diagnostic Study | ✅ Complete (negative result) |
| 5 | AGM + Ablations (8 seeds) | ✅ Complete |
| 6 | Qualitative Token Analysis | ✅ Complete |
| 7 | Statistical Tests (bootstrap CIs) | ✅ Complete |
| 8 | Paper Writing (Mask-only as main method) | ✅ Complete |
| 9 | arXiv Preprint | ✅ Live — [2605.03091](https://arxiv.org/abs/2605.03091) |
| 10 | Submission to BlackboxNLP 2026 | ✅ Submitted (July 17 deadline) |
---

## Notes for Teammates

- **Do not push checkpoints** — large files, gitignored
- **All baselines and AGM use 8 seeds** — [42, 43, 44, 45, 46, 47, 48, 49]
- **AGM batch size is 4 (effective 8)** — due to double backward memory requirement
- **IRM lambda must be 1e2** — 1e4 causes model collapse
- **IRM early stopping must use the source-domain val loader**, not the target loader — see fix note above
- **DRO needs worst-case early stopping** — use min(val_f1) across domains, not average
- **Fish uses Reptile-style inner/outer loop** — not simple gradient averaging
- **Baselines save no checkpoints** — metrics only, best weights in memory
- **Log everything to W&B from run 1**
- **During double-blind review (~July 17 – Sept 8), avoid public promotion** (e.g. LinkedIn posts) referencing the paper or its results

---

## Citation

If you use this work, please cite the arXiv preprint:

```
@misc{harkare2026agm,
  title={Attribution-Guided Masking for Robust Cross-Domain Sentiment Classification},
  author={Harkare, Shubham and Kulkarni, Yash and Yogesh Babu, Arvind Suresh},
  year={2026},
  eprint={2605.03091},
  archivePrefix={arXiv}
}
```

---

## References

- Sagawa et al. (2020). *Distributionally Robust Neural Networks for Group Shifts.* ICLR.
- Shi et al. (2022). *Gradient Matching for Domain Generalization.* ICLR.
- Ganin et al. (2016). *Domain-Adversarial Training of Neural Networks.* JMLR.
- Arjovsky et al. (2019). *Invariant Risk Minimization.* arXiv.
- Sundararajan et al. (2017). *Axiomatic Attribution for Deep Networks.* ICML.
- Ross et al. (2017). *Right for the Right Reasons.* IJCAI.
- Rosenfeld et al. (2021). *The Risks of Invariant Risk Minimization.* ICLR.