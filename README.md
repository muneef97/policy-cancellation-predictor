# Churn Busters — HBF Proactive Retention Challenge

**WA Data Science Innovation Hub Health Hackathon 2026**
Team: Muneef Muhammed, Sandra Elsa Dennies, Diya Susan Eapen

## The challenge

HBF, a not-for-profit Australian health insurer, asked: which policies are about to cancel, and what should we actually do about each one? The brief posed four questions — what predicts cancellation, how granular can that prediction get, how early can we catch it, and what intervention fits each case — against 24 months of real, de-identified policy history.

## What we built

A LightGBM model that scores every active policy for its 3-month cancellation risk, paired with a cost-benefit layer that turns that score into a recommended action (phone call, digital nudge, with or without a discount) using HBF's own stated formula and economics.

Headline results, all on held-out validation data:

- **97.6% ROC-AUC**, catching **95.2%** of policies that actually cancel within 3 months by contacting just the riskiest 5% of the book
- **$14.66M/month** in expected net value if the resulting contact plan is acted on today, across 538,738 active policies
- One signal — an unresolved marketing sign-up status — is **20 to 40 times** stronger than any other feature, across every cancellation type
- WA-based policyholders churn at **1.51%** over 3 months versus **3.76%** everywhere else — a gap that survives controlling for acquisition channel, income tier, and tenure

### Why a 3-month window, specifically

We trained and validated the model on every window from 1 to 9 months before picking one, rather than assuming. Raw accuracy (PR-AUC) actually keeps climbing the further out you look — but that's mostly because the definition of "positive" gets broader (the base rate roughly triples from 3 to 9 months), not because the model gets fundamentally sharper. ROC-AUC, which isn't skewed by base rate, tells the fairer story: it climbs steeply to 3 months (96.1% → 97.6%) then plateaus (97.6–97.9% all the way to 9 months). 3 months is the point where accuracy has already leveled off while the flagged share of the book is still tight enough to act on — and short enough to front-run the book's biggest seasonal event, a March premium-reset spike in cancellations. 1 month, by contrast, is the one window where our model actually loses to a simple baseline, because it forces the model to pinpoint an exact triggering month rather than rank overall risk.

## Repository structure

```
scripts/
  01_eda.py              exploratory analysis of the raw policy history
  02_build_label.py      builds the cancellation-within-N-months label per policy-month
  03_split.py            train/validation split, with leak-safety exclusions
  04_build_features.py   feature engineering / design matrix construction
  05_train_baseline.py   logistic regression baseline, all 7 prediction windows
  06_train_lightgbm.py   LightGBM model, SHAP analysis, per-type evaluation
  07_cost_benefit.py     HBF's cost-benefit formula applied to the live book
  09_policy_explorer.py  builds the individual-policy explorer data for the deck
  10_type_drivers.py     per-cancellation-type feature driver summaries

models/
  lightgbm_{1,2,3,4,5,6,9}m.txt       trained LightGBM models, one per window
  baseline_logreg_{1,2,3,4,5,6,9}m.pkl logistic regression baselines, one per window

presentation/
  PITCH_INSIGHTS.md        running log of every finding behind the pitch, with sources
  shap_summary_3m.png      SHAP feature importance, 3-month model
  policy_explorer_3m.json  individual policy records used in the deck's explorer
  type_drivers_3m.json     per-cancellation-type driver data

HBF_Retention_Pitch_Deck_Theme.html   interactive, single-file HTML pitch deck
```

## Assumptions and limitations

Full detail is in `presentation/PITCH_INSIGHTS.md`. In short: `value_retained` ($1,000), phone cost ($20), digital cost ($0.10), and the 12%-of-annual-premium discount cap are all HBF's own stated figures, not ours. Intervention success rate is genuinely unknown — HBF said so explicitly — so it's swept across 10/20/30% scenarios rather than asserted as fact. Two additional multipliers (phone being more persuasive than digital; a discount lifting success rate) are our own disclosed assumptions, needed because the literal formula otherwise makes those two options mathematically unable to ever be chosen.

## Data

Raw policy data is not included in this repository (privacy, and it isn't ours to redistribute). Everything here — scripts, trained models, presentation assets — reflects the pipeline run against HBF's de-identified dataset for this hackathon.
