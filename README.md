# Churn Busters: predicting HBF policy cancellations

**WA Data Science Innovation Hub Hackathon 2026, HBF Challenge 1: Proactive Retention**
Team: Muneef Muhammed, Sandra Elsa Dennies, Diya Susan Eapen

Every month, the model gives each active policy a chance of cancelling within the next 3 months, shows what is
driving that chance, and picks the most cost-effective retention action using HBF's own formula.
No HBF data is included in this folder.

---

## 1. What is in this folder

| Path | What it is | When to use it |
|---|---|---|
| `README.md` | This guide | Start here |
| `live_demo/HBF_Retention_Hub.html` | Interactive tool. Opens in any browser, works offline | Browse the at-risk list, see why a policy is flagged, try what-if changes, re-cost the plan with your own assumptions |
| `models/final/model.txt` | **The main model**: chance a policy cancels within 3 months (LightGBM, text format) | Scoring policies |
| `models/final/meta.json` | Its settings: features used, training months, class weight used for calibration | Scoring policies, checking setup |
| `models/final/metrics.json` | Its test results on Dec 2025 to Jan 2026 | Checking performance |
| `models/category_codes.json` | How each text category (state, tier, ...) is turned into a number | Needed to score new data the same way |
| `models/type_*/model.txt` | One model per cancellation type (Switcher, Leave PHI, System Lapsed, Non-Recoverable) | Question 1 |
| `models/exact_month_1/2/3/` | Check models: "leaves exactly 1, 2 or 3 months from now" | Question 3 cross-check |
| `results/` | Every result, as JSON or CSV (see the table below) | Looking up any number |
| `code/` | The full pipeline, from HBF's raw extracts to the results and live demo | Re-running or changing it |

**The results files, and which question each one answers**

| File | What it holds |
|---|---|
| `final_check.json` | Performance on the locked Mar to May 2026 months, scored once at the very end |
| `group_check.json`, `group_check.csv` | Does the model work equally well for every state, age, tenure, channel and income tier? |
| `type_drivers.json` | **Question 1**: top warning signs for each cancellation type, with the real cancellation rate of every group |
| `granularity.json` | **Question 2**: how well the type, and the detailed reason, can be told apart |
| `how_early.json` | **Question 3**: share of each type's leavers already flagged 3, 2 and 1 months before they left |
| `risk_journeys.json`, `.parquet` | Question 3 extra: how the risk of individual leavers moved in the six months before they left |
| `retention_plan.json` | **Question 4**: calibration check, the best action for every policy, and what the plan is worth |
| `excluded_fields.json` | Evidence for the fields we left out (see section 6) |
| `summary_data.json` | The summary numbers the live demo displays |

---

## 2. How the pipeline works

`code/run_all.sh` runs these steps in order. Each script starts with a plain-English header saying what it
does, what it reads and what it writes, so they can be read top to bottom like chapters.

| Step | Script | What happens |
|---|---|---|
| 1 | `01_build_labels.py` | The target: does each policy cancel within 3 months of each month-end, and if so how and why |
| 2 | `02_build_features.py` | Features: claims history, survey answers, policy details and recent changes |
| 3 | `03_prepare_model_data.py` | Joins labels and features and stores one file per month |
| 4 | `04_train_models.py` | Trains the main model and the exact-month check models |
| 5 | `05_final_check.py` | Scores the locked Mar to May 2026 months, once |
| 6 | `06_group_check.py` | Fairness check across groups |
| 7 | `07_type_models.py` | Question 1 |
| 8 | `08_granularity.py` | Question 2 |
| 9 to 10 | `09_how_early.py`, `10_risk_journeys.py` | Question 3 |
| 11 | `11_retention_plan.py` | Question 4 |
| 12 to 13 | `12_summary_data.py`, `13_build_live_demo.py` | Builds the live demo |

Shared pieces: `config.py` holds **every setting and assumption in one place** (costs, success rates, months,
feature lists); `common.py` holds the shared helpers; `templates/` holds the live demo's page.

---

## 3. Re-running it

You need Python 3.10 or newer and about 10 GB of free disk space.

```bash
cd code
pip install -r requirements.txt
./run_all.sh --data /path/to/HBF/data      # all 13 steps, about 30 to 60 minutes
./run_all.sh --list                        # show the steps
./run_all.sh --data ... --from 11          # re-run from step 11, e.g. after changing an assumption in config.py
```

- The data folder must contain HBF's four extracts as supplied: `DATA_Monthly_Active_Policy_Spline/`,
  `DATA_Cancellations/`, `DATA_Claims/` and `DATA_Channel Survey Feedback/`.
- Large intermediate files go to `work/` and every run is logged to `logs/`.
- **The run is reproducible**: data is read in a fixed order and every model uses fixed seeds. We checked that
  two different machines give identical results. The package versions we used are listed in `requirements.txt`.
- On a smaller machine, set `HBF_DUCKDB_MEMORY=2GB` (default 3GB) before running.
- On Anaconda you may see warnings about `numexpr` or `bottleneck`; they are harmless. To silence them:
  `conda install -c conda-forge "numexpr>=2.10" "bottleneck>=1.4"`.

---

## 4. Headline results

Trained on Sep 2024 to Oct 2025, tested on Dec 2025 to Jan 2026 (1,077,079 policy-months the model never
saw, 1.85% of which cancelled within 3 months), then checked once on the locked Mar to May 2026 months.

| Measure | Dec 2025 to Jan 2026 | Mar to May 2026 (locked) |
|---|---|---|
| Ranking skill (ROC-AUC): how often a random leaver is ranked above a random stayer | 73.7% | 76.1% |
| Share of the riskiest 1% who really cancel | 20.5% (11x the average) | 19.6% (12x) |
| Share of all leavers found in the riskiest 5% | 25.2% | 29.0% |

Fairness: the model works about equally well in every state (ranking 70% to 74%). It is
weaker for younger members (about 68% for ages 18 to 34) than older members
(about 80% for 65+).

---

## 5. HBF's four questions

### 1. Which variables are most predictive of each cancellation type?
One model per type (`models/type_*`). The top 3 warning signs of each, ranked by two independent methods
(the model's own explanation, SHAP, and a shuffle test):

| Type | Top 3 warning signs (riskiest group vs safest group) | Ranking skill |
|---|---|---|
| Switcher | **Time as a member** (under 3 months: 10x as likely as 20+ years); **State** (NSW: 4x as likely as WA); **Monthly premium** ($121 to $211/month: 2x as likely as $525 to $1,689/month) | 75% |
| Leave PHI | **Monthly premium** ($1 to $120/month: 14x as likely as $525 to $1,689/month); **Time as a member** (under 3 months: 23x as likely as 20+ years); **Type of cover** (no hospital cover: 3x as likely as hospital cover) | 82% |
| System Lapsed | **Time as a member** (under 3 months: 21x as likely as 20+ years); **How they pay** (paying at a branch: 7x as likely as direct debit); **Type of cover** (no hospital cover: 3x as likely as hospital cover) | 85% |
| Non-Recoverable | **Monthly premium** ($211 to $335/month: 18x as likely as $525 to $1,689/month); **Age** (aged 75+: 10x as likely as aged 45-54); **Single or family** (singles: 16x as likely as families) | 88% |

Time as a member matters for most types: new members leave more for any reason. What separates the types is
what comes next: state and price for Switchers, price for Leave PHI, how people pay for System Lapsed, and age
and household for Non-Recoverable. "Type of cover" comes from the Lifetime Health Cover loading, which is blank
when hospital cover doesn't apply (extras-only, ambulance-only or overseas visitor cover).

### 2. How granular can we predict?
- **Which type:** among leavers, the model names the type correctly 66% of the time
  (always guessing the most common type: 48%). Switcher, System Lapsed and
  Non-Recoverable separate well; Leave PHI is the hardest, as it is most often mistaken for Switcher.
- **Detailed reason:** works only for Non-Recoverable, e.g. deceased vs moving overseas (77%
  correct vs 42% by guessing). Leave PHI reasons are only
  weakly separable (41% vs 32%).
  For Switchers, 79% of records only say "Request received from other fund", which records how they left, not why.
- So: predict down to the **type**, and capture the reason at exit to go further.

### 3. How early can each type be predicted?
Share of each type's leavers already in the riskiest 5% of the book, 3, 2 and 1 months before they left:

| Type | 3 months before | 2 months before | 1 month before |
|---|---|---|---|
| Switcher | 12% | 11% | 27% |
| Leave PHI | 25% | 27% | 38% |
| System Lapsed | 28% | 31% | 36% |
| Non-Recoverable | 28% | 38% | 60% |
| All leavers | 19% | 22% | 36% |

System Lapsed and Leave PHI show up two to three months ahead; Switchers mostly in the final month. So the
book should be scored **every month**.

### 4. What interventions suit each group?
Every policy gets the option worth the most under HBF's formula:
**worth = chance of leaving x success rate x ($1,000 - discount) - contact cost**

| Action | Who gets it | Policies (31 Jan 2026) |
|---|---|---|
| Phone call ($20) | Highest risk: about 1 in 5 chance of leaving or more | 1,496 |
| Digital message + 12% discount | Everyone else paying under about $160 a month | 161,648 |
| Digital message (10 cents) | Everyone else | 373,168 |
| No action | Almost no chance of leaving (under 1 in 2,000) | 2,426 |

Worked example, a member paying $150 a month with a 10% chance of leaving (a 12% discount is $216 a year):
digital $19.90, **digital + discount $20.28** (best), phone $10.00, phone + discount $10.58.
In plain words: for 100 members like this, 10 would leave; a message keeps 2 of them ($2,000) for $10 of messages.

Across all 538,738 policies the plan is worth about $1.79M a month at a 20%
success rate ($0.86M at 10%, $2.74M at 30%).
Non-Recoverable leavers are worth about $2 each to chase, so they get no paid offers.

**Our recommendation**

| Type | Focus | What to do | When to act |
|---|---|---|---|
| Switcher | 1 | Offer before the 1 April price rise | Mostly visible in the final month: act fast |
| System Lapsed | 2 | Move branch payers to direct debit | Visible 2 to 3 months ahead: plan contact early |
| Leave PHI | 3 | Value check-in in the first months of membership | Visible 2 to 3 months ahead: plan contact early |
| Non-Recoverable | - | No paid offers | - |

Score every policy monthly, start by contacting the riskiest 20% (about half of the members likely to leave),
keep calls for the riskiest few, and test on a small group before rolling out.

---

## 6. Choices worth knowing

- **Target:** "cancels within 3 months", from each month-end snapshot. System Lapsed cancellations are dated by
  when cover ended; all others by when HBF processed them.
- **Fields left out on purpose** (`results/excluded_fields.json`):
  `MARKETING_OPTIN` never changes for a policy, yet "Unknown" almost always ends in a cancellation, so it seems to
  be filled in after the fact; `INCOME_TIER` is blanked while a cancellation is processed; this month's
  `ARREARS_FLAG` already triggers HBF's own process (the flag from 2 months earlier is kept). If HBF confirms
  opt-in is captured at sign-up, it could go back in.
- **Honest testing:** months are split in date order with a one-month gap: train, gap, test, gap, locked final
  check. The locked months were scored once, at the end.
- **Calibration:** leavers are weighted up during training, and scores are converted back to real-world chances.
  On the test months the model predicted 1.57% on average against 1.85%
  actual, so if anything it slightly underestimates.
- **Assumptions we added** (all in `code/config.py`): HBF gave no success rate, so we used 20% for a digital
  message and tested 10% and 30%; a call is assumed 1.5x and a discount 1.3x as persuasive.
