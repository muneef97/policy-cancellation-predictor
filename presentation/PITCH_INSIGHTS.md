# Pitch-worthy insights log
Running list of the strongest findings from the pipeline, kept as we go so nothing
gets lost before the presentation. Each one is backed by a real number we computed,
not a guess -- cite the script that produced it if asked to defend it.

## 1. The "70-day rule" shows up in the data on its own
The challenge pack states System Lapsed cover "ends after 70 days of missed payments."
We didn't take that on faith -- we found it independently in the Cancellations table:
the median gap between the effective cease date and the processed cease date for
System Lapsed is **-78 days**, essentially matching the documented rule, while every
other cancellation type shows only a 3-9 day gap. This single number is what justified
using a *type-specific* anchor date for the label (effective date for System Lapsed,
processed date for everything else) rather than blindly copying HBF's own example SQL,
which would have taught the model a systematically-delayed version of lapse behaviour.
(Source: scripts/01_eda.py, scripts/02_build_label.py)

## 2. A scary-looking data artifact that turned out to be self-solving
1,551 cancellation records had effective dates over a year before their processed
date -- some backdated by nearly 11 years. Rather than assume this was noise to patch
around, we traced it: only 65 of those 1,284 System-Lapsed cases even appear in the
spine's 2024-2026 window, and the join logic (only count a cancellation if it's in the
*future* relative to an observation row) mechanically excludes the rest, since a
years-old date can never be "after" a 2024+ snapshot. A structural property of the
join design solved a problem we thought we'd need to patch by hand.
(Source: scripts/01_eda.py)

## 3. "Degenerate tail months" -- a leak-safety subtlety most teams will miss
Near the end of the data window, a month can only retain CONFIRMED positives (a
negative can't be verified without a full future window) -- which makes those trailing
months 100% positive and actively misleading if left in training or evaluation. We
caught this by inspecting month-by-month positive rates (not just trusting the label
script ran without error) and added an explicit exclusion step. This is exactly the
kind of subtle correctness issue a rigorous judge would probe for, and it's also why
insight #9 below uses the *post-exclusion* positive rates, not the raw ones.
(Source: scripts/03_split.py)

## 4. Seasonality: churn is NOT flat across the year, and it's type-specific
Switcher cancellations spike hard every March/April and July (e.g. April 2025: 1,792
vs a ~1,000-1,200 baseline) -- lining up with Australian PHI premium increases and
EOFY competitor offers. System Lapsed and Non-Recoverable stay flat month to month.
This is direct, data-backed evidence for Q3 (timing): retention effort against
Switcher risk should be seasonally weighted, not spread evenly across the year.
(Source: scripts/01_eda.py)

## 5. Geography: a 2-3x churn gap between WA and everywhere else, and it is NOT
## explained by who those customers are
WA policies churn at 1.36% (3-month window) vs 2.7-4.4% in every other state --
roughly 2-3x higher outside WA. The first version of this finding was just the raw
gap; we then stress-tested it by checking whether it's really just a stand-in for
channel mix, income tier, or tenure differences between WA and interstate customers.
It isn't -- the WA discount survives in every slice we checked:
  - By acquisition channel: WA is lower in 6 of 7 channels (e.g. Digital: 2.26% WA
    vs 4.41% other; Branch: 1.18% vs 2.77%; Legacy: 0.90% vs 1.37%). The one
    exception, Broker, has only 3,225 WA policies -- too small to read into.
  - By income tier: WA is lower in every tier, including the missing-tier group
    (11.53% WA vs 19.78% other -- see insight #6, the WA discount applies there too).
  - By tenure band: WA is lower in every band, from brand-new (0-3m: 5.68% WA vs
    10.56% other) to 20+ year veterans (0.76% vs 1.23%).
HBF's book is 85% WA / 15% interstate. Since the gap holds after controlling for
channel, income, and tenure, this points to something structural about being a
WA-based customer of a WA-based fund (network coverage, brand trust, community ties)
rather than a demographic artifact -- which makes interstate growth a genuine,
quantified retention risk worth its own strategy, not just "more of the same at a
smaller scale."
(Source: DuckDB analysis against DATA_derived/model_data_3m.parquet, this conversation)

## 6. The missing-value that's actually the strongest signal we've found
Policies with a MISSING `INCOME_TIER` churn at 11-16%, vs 1.4-2% for every populated
tier -- close to a 10x difference, and it holds inside both WA and interstate (see
insight #5). Likely story: an unresolved income tier means the policy hasn't fully
"settled in" (rebate paperwork incomplete), and that same unsettled status predicts
early departure. Implication for the model: this must be kept as an explicit "missing"
category, never silently imputed away, or the model loses its best single signal.
(Source: DuckDB balance-check, this conversation)

## 7. Some customers cancel and come back -- repeatedly
2,722 policies show more than one cancellation record (up to 14 for a single policy
over ~19 months), with a median gap of 165 days between events -- these are real
reinstate/re-lapse cycles, not duplicate data. Worth a callout as its own behavioural
segment ("serial lapsers") that might need a different intervention than a first-time
canceller.
(Source: scripts/01_eda.py)

## 8. Age and channel both show clean, textbook-shaped churn curves
Churn declines steadily with age from 18-24 (~3.5%) down to a low around 55-64
(~1.1%), then ticks back up for 75+. Digitally-acquired customers churn ~3x more than
Legacy-acquired ones (3.0% vs 0.96%). Both are clean enough patterns to visualize
directly and instantly legible to a non-technical judge.
(Source: DuckDB balance-check, this conversation)

## 9. Class imbalance grows steadily with prediction window (corrected numbers)
Using the post-degenerate-tail-exclusion positive rate (the honest, leak-safe number
-- see insight #3), positive rate goes from 0.56% (1-month window) to 4.67%
(9-month window):

| window | 1m | 2m | 3m | 4m | 5m | 6m | 9m |
|---|---|---|---|---|---|---|---|
| positive rate | 0.56% | 1.12% | 1.67% | 2.20% | 2.73% | 3.23% | 4.67% |

That's a remarkably steady ~0.5 percentage-point increase per added month, with no
discontinuities -- a clean sanity check that the label construction has no hidden
bugs, and a good one-slide visual for explaining *why* longer-horizon prediction is a
harder, noisier problem. (Note: an earlier draft of this insight cited 5.84% for the
9-month window -- that was the *pre*-exclusion number, inflated by exactly the
degenerate-tail artifact in insight #3. Caught and fixed before this went anywhere
near a slide.)
(Source: scripts/02_build_label.py, scripts/03_split.py)

## 10. First working model: catch 95% of cancellers by contacting just 5% of the book
LightGBM on the 3-month window (val set): ROC-AUC 0.976, PR-AUC 0.469 against a
1.85% base rate (roughly 25x better than random). The operating-point numbers are
the ones a non-technical judge will actually feel: if HBF's retention team can only
proactively contact the top 5% highest-risk policies each month, this model catches
95.2% of everyone who will actually cancel within 3 months, at 35% precision (1 in
~3 contacted would genuinely have left). Contacting the top 1% alone still catches
26% of cancellers at 49% precision -- useful for a smaller, higher-touch outreach
tier. A simple Logistic Regression baseline on the same data only reached ROC-AUC
0.972 / PR-AUC 0.360 -- LightGBM's ability to use categoricals and missing-value
patterns natively (no one-hot, no imputation) is a real, measured improvement, not
just a fancier label. Both numbers are the ones confirmed reproducible in insight #12 -- rebuilt the underlying dataset from scratch and retrained/refit twice, and these are what came back both times (LightGBM shows a tiny residual sensitivity to rebuild order in the 3rd decimal place, from its row-position-based bagging -- negligible next to the reproducibility bug we actually found and fixed).
(Source: scripts/06_train_lightgbm.py)

## 11. The WA-vs-interstate design question, settled with evidence, not a guess
Before committing to one unified model, we checked whether HBF's book should really
be modelled as one population, given insight #5's finding that WA churns 2-3x
lower than interstate. The unified LightGBM model performs strongly on BOTH
segments -- WA: ROC-AUC 0.976 (95.0% recall at top 5% contacted); interstate:
ROC-AUC 0.968, and actually a *higher* PR-AUC (0.492 vs 0.454), because interstate's
higher base rate (3.76% vs 1.51%) gives the model more positive examples per
customer contacted to learn from. That's direct evidence that letting the model see
STATE_CODE as a feature (rather than building two separate models) was the right
call -- a tree-based model can already learn different rules for each segment
inside one model, and splitting the data would have thrown away nearly a third of
all cancellation examples (see insight #5's underlying numbers) for no measured gain.
(Source: scripts/06_train_lightgbm.py)

## 12. Caught our own reproducibility bug before it could embarrass us
Our first baseline runs gave different ROC-AUC/PR-AUC numbers on identical code and
identical data (0.352 vs 0.255 PR-AUC at one point) -- a red flag most teams would
either not notice or paper over. We chased it down through several real layers: a
non-deterministic streaming SGD classifier (replaced with a proper single-batch fit),
a DuckDB reservoir sample that turned out to depend on physical row order rather
than data content (replaced with a hash-of-policy-ID filter, which is invariant to
how any given file rebuild happens to lay rows out on disk), and a memory bug where
an unnecessary DataFrame copy was pushing training out of memory on a constrained
machine (fixed by editing in place instead of copying). We verified the fix
properly, not just by eyeballing one run: two independent full rebuilds of the
underlying dataset from scratch now produce bit-for-bit identical results --
ROC-AUC 0.9723, PR-AUC 0.3603 on the 3-month baseline. This kind of reproducibility
discipline is exactly what a real production deployment needs and what a rushed
hackathon model usually skips.
(Source: scripts/05_train_baseline.py, this conversation)

## 13. SHAP confirms the dominant driver is real -- and checked it isn't a leak
LightGBM's raw gain-based importance showed MARKETING_OPTIN at >3x the next feature,
which is exactly the kind of result that deserves suspicion rather than a victory
lap. SHAP confirms it's a genuine, well-behaved effect (not a gain-metric artifact):
policies with an "Unknown" marketing opt-in status (U) churn at 37.84% within 3
months, vs 0.08% for opted-in (Y) and 0.20% for opted-out (N) -- a 180x+ gap, and
not a tiny group either (26,661 policies, ~4.9% of the book in the val split alone).
Before treating this as a headline number, we checked it for leakage: is "U" a
status that flips right before someone cancels, or a genuine longstanding trait?
Checked directly -- every single policy in the dataset (608,088 of them) shows
exactly ONE distinct MARKETING_OPTIN value across its ENTIRE observed history, and
for policies that churned while showing "U", 100% of their monthly rows show "U"
going back to their first appearance in the data. It's set once (almost certainly
at signup) and never changes -- a real, stable customer trait, not a leak. Likely
the same underlying story as insight #6 (missing INCOME_TIER): an incomplete
onboarding signal that correlates with a customer who never fully "settled in".
SHAP also resolved the baseline's counter-intuitive negative coefficient on
NEGATIVE_SENTIMENT_CALLS_LAST_6M -- the tree model shows the intuitive POSITIVE
relationship (correlation +0.58 between call count and predicted risk; policies
with 3+ negative-sentiment calls in 6 months show meaningfully higher predicted
risk), suggesting the linear baseline's sign flip was a multicollinearity artifact
the tree model doesn't share.
(Source: scripts/06_train_lightgbm.py --step shap, this conversation)

## 14. All 7 prediction windows run: risk sharpens the further out you look, up to a point
Ran the full pipeline (build, baseline, LightGBM) on every window HBF might care about
-- 1, 2, 3, 4, 5, 6 and 9 months ahead. Two clear, honest findings, not just a win story:

| window | base rate | Baseline ROC/PR | LightGBM ROC/PR |
|---|---|---|---|
| 1m | 0.62% | 0.970 / 0.168 | 0.961 / 0.133 |
| 2m | 1.18% | 0.969 / 0.251 | 0.972 / 0.351 |
| 3m | 1.85% | 0.972 / 0.360 | 0.976 / 0.469 |
| 4m | 2.31% | 0.972 / 0.418 | 0.976 / 0.523 |
| 5m | 2.77% | 0.971 / 0.465 | 0.976 / 0.571 |
| 6m | 3.28% | 0.974 / 0.542 | 0.979 / 0.656 |
| 9m | 4.64% | 0.971 / 0.615 | 0.978 / 0.715 |

First, PR-AUC climbs steadily and substantially the further out the model is allowed
to look (0.133 at 1 month to 0.715 at 9 months) -- a real, honest signal that
cancellation behaviour telegraphs itself well before it happens, and a 6-9 month
window gives the retention team dramatically more useful risk scores than a 1-month
one, not just a "longer runway" number. Second -- and worth being upfront about
rather than only reporting the wins -- LightGBM actually slightly UNDERPERFORMS the
simple baseline at the 1-month window (0.133 vs 0.168 PR-AUC), the only window where
that happens. Checked why rather than shrugging it off: val PR-AUC plateaus by round
~40-60 and val ROC-AUC actually peaks around round 60 then mildly degrades -- with
only ~45,000 positive training examples at this window's 0.56% base rate (the
fewest of any window), the tree ensemble has too little positive signal to
outperform a heavily-regularized linear model, and likely starts fitting noise
instead. A useful, honest nuance for a judge who asks "is more complexity always
better?" -- no, and here's the one place in our own results that proves it.
(Source: scripts/05_train_baseline.py, scripts/06_train_lightgbm.py, this conversation)

## 15. One risk score, four different cancellation stories underneath it
Broke the 3-month LightGBM model's performance down by WHY people actually cancelled
(Switcher, System Lapsed, Non-Recoverable, Leave PHI), using the SAME unified risk
score HBF's retention team would actually use -- not a separate model per type. At a
realistic 5% contact capacity, the model catches 99%+ of Switcher, Non-Recoverable
and Leave PHI cancellers, but only 86.1% of System Lapsed -- these are more likely
arrears-driven, possibly quieter/less-signalled departures than an active switch to
a competitor. More striking: at the very top of the ranking (1% contact capacity --
the highest-confidence tier), Non-Recoverable cancellers are disproportionately
represented (42.2% already caught) while Switchers -- the single LARGEST cancelling
group at 44% of all val-set cancellations -- are comparatively spread out (only
20.2% caught this early, despite 99.6% eventually caught by 5%). Practical read for
HBF: a very-limited-capacity outreach list (top 1%) will skew toward
Non-Recoverable/System Lapsed cases, while catching the bulk of Switcher risk needs
the fuller 5% contact list. Worth shaping the retention team's messaging by which
type dominates a given contact tier, not treating "high risk" as one undifferentiated
bucket.
(Source: scripts/06_train_lightgbm.py --step eval_by_type, this conversation)

## To fold in once available
- SHAP breakdown PER CANCELLATION TYPE specifically (done: overall SHAP: insight #13) (Q1)
- Expected-net-value intervention economics using HBF's own formula (done: insight #16) (Q4)

## 16. Cost-benefit: two "levers" in HBF's own formula would never fire without saying so out loud
Implemented HBF's official Q4 formula (`churn_prob x success_rate x (value_retained -
incentive_cost) - contact_cost`) with its official assumptions ($1,000 average value
saved, $20 phone / $0.10 digital contact cost, discounts up to 12% of annual premium)
against the live 3-month model on the most recent month in val (538,738 active
policies). Two of HBF's own four intervention options -- phone contact and discounted
offers -- turned out to be **mathematically incapable of ever being chosen** under the
formula's literal reading: phone only ever costs more than digital for the same
success rate (strictly dominated), and a discount only ever *reduces* value_retained
with no compensating change to success rate (also strictly dominated). The first full
run proved this isn't hypothetical -- it actually happened: 0% phone, 0% discount,
100% digital-no-discount, every time, across every success-rate scenario tested. That
would have meant reporting "our optimal plan never calls anyone and never discounts
anything" as if it were a finding, when it's actually a gap in the formula as literally
read. Fixed by adding two disclosed (non-HBF, clearly labeled) assumptions: phone
success rate = digital rate x a multiplier (baseline 1.5x, tested 1.25x-2.0x, since a
personal two-way call should out-convert a passive digital nudge), and discount
success rate = no-discount rate x a multiplier (baseline 1.3x, tested 1.15x-1.5x,
since paying money should buy some persuasion). With that, the plan becomes a genuine
four-way decision: at baseline assumptions, **64.0% digital-no-discount, 25.0%
digital-discount, 5.8% phone-no-discount, 5.2% phone-discount**, for a total expected
net value of **$14.66M this month** across the active book -- and every option is used
somewhere.
The discount break-even math is its own finding: a discount only pays for itself when
it lifts the success rate by more than `1000 / (1000 - discount_amount)`. At the
book's *average* max discount (12% of a $3,727 average annual premium = $447), that
threshold is **1.81x -- above** our conservative 1.3x baseline assumption. Yet 30.1%
of the book still gets an incentive under that same baseline, because the threshold
scales with premium size: a policy needs less than roughly **$1,923 annual premium
(~$160/month)** for a 12% discount to clear break-even at 1.3x. In other words, this
isn't a contradiction -- it's a genuine, data-derived targeting rule: **discounts pay
off on lower-value policies, not the book's highest-premium customers**, which is a
sharper, falsifiable version of HBF's own "match lower-cost activity to lower-value
opportunities" instruction (challenge pack, slide 7).
By cancellation type (retrospective, using type-specific success-rate multipliers
grounded in the challenge pack's own descriptions of each type): Switcher has by far
the highest expected value per policy ($411 mean best option) -- consistent with
"highly contestable with the right, timely offer" -- while Non-Recoverable sits near
break-even ($20, mostly not worth a discount), directly supporting HBF's own guidance
that Non-Recoverable cases should generally be excluded from paid intervention
targeting.
Two additional bugs caught and fixed during this step before any number was trusted:
~0.26% of policies had a NULL premium that silently turned every downstream total
into `$nan` (fixed by reusing the same train-median imputation already validated in
the baseline pipeline, not inventing a new one), and a duplicate-column collision
when re-selecting MONTHLY_PREMIUM for its raw dollar value alongside its already-
selected standardized model feature (fixed with an explicit SQL alias).
(Source: scripts/07_cost_benefit.py, this conversation)
