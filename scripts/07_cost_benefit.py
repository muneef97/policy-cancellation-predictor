"""
Step 7 - Cost-benefit / intervention layer, 3-month window.

Implements HBF's own Q4 formula from the challenge pack, not an invented one:

    Expected net value = churn_probability
                          x intervention_success_rate
                          x (value_retained - incentive_cost)
                          - contact_cost

Official assumptions from the challenge pack (CHALLENGE_INFORMATION/HBF Challenge
Pack.pptx, slide 7):
  - average policy saved is worth $1,000 (value_retained)
  - phone communication costs $20 (contact_cost, phone channel)
  - digital communication costs $0.10 (contact_cost, digital channel)
  - discounts up to 12% of ANNUAL premium may be applied, which reduces the value
    being saved by an equal amount (incentive_cost = up to 0.12 x annual_premium,
    and value_retained is reduced by exactly that in the formula)
  - intervention_success_rate is explicitly NOT given ("you are not expected to
    know the true effectiveness of each intervention... clearly state your
    assumptions, test reasonable scenarios") -- treated as a sensitivity parameter
    throughout, never a single hardcoded number presented as fact.

Per policy, evaluates FOUR intervention options (digital/phone x with/without a
12%-of-annual-premium discount) and picks whichever has the highest expected net
value (or no action, if all are negative).

TWO disclosed, non-HBF assumptions make this a genuine four-way decision instead
of a foregone one -- both documented in detail where they are defined below:
  1. PHONE_EFFECTIVENESS_MULTIPLIER -- phone must be more effective than digital,
     or it is strictly dominated (costs more, same success rate) and never fires.
  2. DISCOUNT_SUCCESS_MULTIPLIER -- a discount must lift the success rate, or it
     is strictly dominated too (same success rate, but reduces value_retained)
     and never fires. Without this, HBF's own "discounts up to 12%" lever would
     be mathematically incapable of ever being the best option, in any scenario.

Run:
    python3 scripts/07_cost_benefit.py --window 3
"""
import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import importlib
baseline_mod = importlib.import_module("05_train_baseline")
lgb_mod = importlib.import_module("06_train_lightgbm")

OUT_DIR = "DATA_derived"
MODEL_DIR = "models"
PRESENTATION_DIR = "presentation"

# ---- Official HBF assumptions (challenge pack, slide 7) --------------------
VALUE_RETAINED_FLAT = 1000.0     # "an average policy saved is worth $1,000"
PHONE_CONTACT_COST = 20.0        # "a phone communication costs $20"
DIGITAL_CONTACT_COST = 0.10      # "a digital communication costs 10 cents"
MAX_DISCOUNT_PCT_OF_ANNUAL = 0.12  # "discounts... up to 12% of annual premium"

# intervention_success_rate is NOT given by HBF -- this is a sensitivity grid,
# not an assumed truth. The example in the Q4 slide itself uses 10% and 20%.
SUCCESS_RATE_SCENARIOS = [0.10, 0.20, 0.30]
BASELINE_SUCCESS_RATE = 0.20  # the "central" scenario, applied to the DIGITAL channel, no discount

# Additional assumption 1 (ours, not HBF's -- disclosed, not hidden): a phone
# call is more effective than a digital nudge, since it is personal and
# two-way. If both channels shared one success rate, phone could never win any
# comparison in this formula (it only ever costs more), which would make the
# challenge's own instruction to "reserve calls... for opportunities with
# sufficient expected value" a non-decision -- there would be nothing to
# reserve. A cost/effectiveness tradeoff only exists if phone is assumed more
# effective, so we say so explicitly and test how sensitive the plan is to
# exactly how much more effective (1.25x-2x).
PHONE_EFFECTIVENESS_MULTIPLIER_SCENARIOS = [1.25, 1.5, 2.0]
BASELINE_PHONE_MULTIPLIER = 1.5

# Additional assumption 2 (ours, not HBF's -- disclosed, not hidden): offering a
# discount lifts the success rate above the same channel's no-discount rate.
# HBF's own formula only ever lets a discount REDUCE value_retained -- it never
# touches success_rate -- so without this assumption a discount option is
# strictly dominated by its own no-discount twin (identical odds of success,
# strictly higher cost to the business) and would be mathematically incapable
# of ever being chosen, in any scenario. That would make "discounts up to 12%
# of annual premium," one of HBF's own three named levers (slide 7), a lever
# that can never be pulled -- clearly not the intent. So we assume a discount
# buys some amount of extra persuasion, and test how much that assumption
# matters (1.15x-1.5x). The lower bound of this range (1.15x) is deliberately
# close to the break-even multiplier for an average-size discount (see the
# "Discount break-even" print below) so the sensitivity grid brackets the
# point where the discount lever stops paying for itself.
DISCOUNT_SUCCESS_MULTIPLIER_SCENARIOS = [1.15, 1.3, 1.5]
BASELINE_DISCOUNT_MULTIPLIER = 1.3

# Type-specific success-rate multipliers -- grounded in the challenge pack's OWN
# description of each cancellation type (slide 6: "Types of cancellation"), not
# invented from nothing:
#   - Switcher: "Highly contestable with the right, timely offer" -> above-average
#   - Leave PHI: "Value/affordability levers matter most" -> moderate, offer-sensitive
#   - System Lapsed (Payments Lapsed): "often affordability/disengagement related"
#     -> moderate-low, harder to reach/re-engage
#   - Non-Recoverable: "can't realistically be saved and should generally be
#     excluded from intervention targeting" -> treated as near-zero
# These multiply BASELINE_SUCCESS_RATE for the retrospective by-type analysis
# only (Section 2 below) -- the prospective, deployable calculation (Section 1)
# uses a single uniform rate since we do not have a live TYPE predictor, only a
# binary churn predictor (that is Q1/Q2 territory, not built here).
TYPE_SUCCESS_MULTIPLIER = {
    "Switcher": 1.5,
    "Leave PHI": 1.0,
    "System Lapsed": 0.75,
    "Non-Recoverable": 0.1,
    "(unclassified)": 0.5,
}


def load_val_with_premium_and_type(window_months, cat_code_maps):
    """Same pattern as load_val_with_type() in 06_train_lightgbm.py -- streams
    val split once, capturing raw (unstandardized) MONTHLY_PREMIUM and
    cancellation type alongside the feature array, all row-aligned from a
    SINGLE query. Two independent queries could come back in different row
    orders (the exact lesson from the baseline's reproducibility bug), so
    this must all come from one pass."""
    n_rows = lgb_mod.get_split_count(window_months, "val")
    X = np.empty((n_rows, len(lgb_mod.ALL_FEATURE_COLS)), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.int8)
    premium = np.empty(n_rows, dtype=np.float64)
    cancel_type = np.empty(n_rows, dtype=object)
    info_date = np.empty(n_rows, dtype=object)

    path = f"{OUT_DIR}/model_data_{window_months}m.parquet"
    select_cols = ", ".join(
        [lgb_mod.numeric_select_sql(c) for c in lgb_mod.NUMERIC_COLS] + lgb_mod.CATEGORICAL_COLS
    )
    con = lgb_mod.duckdb_conn()
    # MONTHLY_PREMIUM is aliased to avoid a duplicate-column collision -- it is
    # already one of NUMERIC_COLS (a model feature) and pulled in again here for
    # its raw dollar value, so it needs a distinct name in this query.
    query = f"""
        SELECT label, CANCELLATION_REASON_CLASSIFICATION,
               MONTHLY_PREMIUM AS RAW_PREMIUM,
               INFORMATION_DATE, {select_cols}
        FROM read_parquet('{path}')
        WHERE split = 'val'
    """
    reader = con.execute(query).to_arrow_reader(batch_size=lgb_mod.BATCH_ROWS)

    pos = 0
    for batch in reader:
        df = batch.to_pandas()
        n = len(df)
        for j, c in enumerate(lgb_mod.NUMERIC_COLS):
            X[pos:pos + n, j] = df[c].to_numpy(dtype="float32")
        for j, c in enumerate(lgb_mod.CATEGORICAL_COLS):
            col = df[c]
            if c == "HOSPITAL_EXCESS_AMT":
                col = col.replace("", np.nan)
            codes = col.map(cat_code_maps[c])
            X[pos:pos + n, len(lgb_mod.NUMERIC_COLS) + j] = codes.to_numpy(dtype="float32")
        y[pos:pos + n] = df["label"].to_numpy(dtype="int8")
        premium[pos:pos + n] = df["RAW_PREMIUM"].to_numpy(dtype="float64")
        cancel_type[pos:pos + n] = df["CANCELLATION_REASON_CLASSIFICATION"].fillna("(unclassified)").to_numpy(dtype=object)
        info_date[pos:pos + n] = df["INFORMATION_DATE"].astype(str).to_numpy(dtype=object)
        pos += n
    con.close()
    assert pos == n_rows, f"row count mismatch: filled {pos}, expected {n_rows}"
    return X, y, premium, cancel_type, info_date


def expected_net_value(churn_prob, success_rate, value_retained, incentive_cost, contact_cost):
    return churn_prob * success_rate * (value_retained - incentive_cost) - contact_cost


def run_cost_benefit(window_months):
    print("Loading saved model + metadata...", flush=True)
    booster = __import__("lightgbm").Booster(model_file=f"{MODEL_DIR}/lightgbm_{window_months}m.txt")
    with open(f"{MODEL_DIR}/lightgbm_{window_months}m_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    cat_code_maps = meta["cat_code_maps"]
    best_round = meta["best_round"]

    print("Loading val split (features + premium + cancellation type, row-aligned)...", flush=True)
    X_val, y_val, premium, cancel_type, info_date = load_val_with_premium_and_type(window_months, cat_code_maps)
    print(f"  X_val {X_val.shape}", flush=True)

    churn_prob = booster.predict(X_val, num_iteration=best_round)

    # A small fraction of policies (~0.26% in val) have a NULL MONTHLY_PREMIUM.
    # Rather than let NaN silently propagate through every downstream sum (which
    # is exactly what happened on the first run here -- every total came back
    # NaN), impute with the SAME train-split median already computed and
    # validated for this purpose in the baseline pipeline (compute_medians()),
    # instead of inventing a fresh, inconsistent fallback number.
    medians = baseline_mod.compute_medians(window_months)
    n_missing_premium = int(np.isnan(premium).sum())
    if n_missing_premium:
        print(f"Imputing {n_missing_premium:,} missing MONTHLY_PREMIUM values "
              f"with the train median (${medians['MONTHLY_PREMIUM']:.2f}) "
              f"({n_missing_premium/len(premium)*100:.2f}% of val set)...", flush=True)
        premium = np.where(np.isnan(premium), medians["MONTHLY_PREMIUM"], premium)

    annual_premium = premium * 12.0
    max_discount = annual_premium * MAX_DISCOUNT_PCT_OF_ANNUAL

    # ==========================================================================
    # SECTION 1: Prospective, deployable calculation -- single unified churn
    # model (no type foreknowledge), one flat value_retained ($1,000, per HBF),
    # four channel/discount options per policy, picking the best (or none).
    # This is what HBF could actually run each month on the live book.
    # ==========================================================================
    print("\n" + "=" * 78, flush=True)
    print("SECTION 1: Prospective monthly intervention plan (single most-recent month)", flush=True)
    print("=" * 78, flush=True)

    most_recent_month = max(info_date)
    month_mask = info_date == most_recent_month
    print(f"Using snapshot month: {most_recent_month} ({month_mask.sum():,} active policies)", flush=True)

    cp = churn_prob[month_mask]
    ap = annual_premium[month_mask]
    md = max_discount[month_mask]

    # Discount break-even: a discount option only beats its own no-discount
    # twin when rate_discount * (V - D) > rate_nodiscount * V, i.e. when the
    # discount buys a success-rate MULTIPLIER greater than V / (V - D). Report
    # this using the book's average max discount so the disclosed
    # DISCOUNT_SUCCESS_MULTIPLIER assumption can be judged against it directly.
    avg_discount = float(md.mean())
    breakeven_discount_mult = VALUE_RETAINED_FLAT / (VALUE_RETAINED_FLAT - avg_discount)
    print(f"\n--- Discount break-even (average max discount in book: ${avg_discount:,.2f}) ---", flush=True)
    print(f"    A discount must lift the success rate by more than {(breakeven_discount_mult - 1) * 100:.1f}%"
          f" (i.e. multiplier > {breakeven_discount_mult:.3f}x) to beat the same channel with no discount.", flush=True)
    print(f"    Our assumed baseline multiplier is {BASELINE_DISCOUNT_MULTIPLIER}x"
          f" ({'above' if BASELINE_DISCOUNT_MULTIPLIER > breakeven_discount_mult else 'BELOW'} break-even),"
          f" sensitivity range {DISCOUNT_SUCCESS_MULTIPLIER_SCENARIOS}.", flush=True)

    def build_options(digital_rate, phone_rate, discount_mult):
        return {
            "digital_no_discount": (DIGITAL_CONTACT_COST, np.zeros_like(ap), digital_rate),
            "digital_discount":    (DIGITAL_CONTACT_COST, md, digital_rate * discount_mult),
            "phone_no_discount":   (PHONE_CONTACT_COST, np.zeros_like(ap), phone_rate),
            "phone_discount":      (PHONE_CONTACT_COST, md, phone_rate * discount_mult),
        }

    def scenario_label(digital_rate, phone_rate, discount_mult):
        return (f"digital {digital_rate:.0%}, phone {phone_rate:.0%} "
                f"[{phone_rate / digital_rate:.2g}x], discount mult {discount_mult}x")

    baseline_phone_rate = BASELINE_SUCCESS_RATE * BASELINE_PHONE_MULTIPLIER
    scenarios = [("baseline", BASELINE_SUCCESS_RATE, baseline_phone_rate, BASELINE_DISCOUNT_MULTIPLIER)]
    for r in SUCCESS_RATE_SCENARIOS:
        if r != BASELINE_SUCCESS_RATE:
            scenarios.append((f"success-rate sweep", r, r * BASELINE_PHONE_MULTIPLIER, BASELINE_DISCOUNT_MULTIPLIER))
    for m in PHONE_EFFECTIVENESS_MULTIPLIER_SCENARIOS:
        if m != BASELINE_PHONE_MULTIPLIER:
            scenarios.append((f"phone-multiplier sweep", BASELINE_SUCCESS_RATE, BASELINE_SUCCESS_RATE * m, BASELINE_DISCOUNT_MULTIPLIER))
    for dm in DISCOUNT_SUCCESS_MULTIPLIER_SCENARIOS:
        if dm != BASELINE_DISCOUNT_MULTIPLIER:
            scenarios.append((f"discount-multiplier sweep", BASELINE_SUCCESS_RATE, baseline_phone_rate, dm))

    for tag, digital_rate, phone_rate, discount_mult in scenarios:
        options = build_options(digital_rate, phone_rate, discount_mult)
        print(f"\n--- Scenario ({tag}): {scenario_label(digital_rate, phone_rate, discount_mult)} ---", flush=True)
        nv = {}
        for name, (cost, incentive, rate) in options.items():
            nv[name] = expected_net_value(cp, rate, VALUE_RETAINED_FLAT, incentive, cost)
        nv["no_action"] = np.zeros_like(cp)

        stacked = np.column_stack([nv[k] for k in options.keys()] + [nv["no_action"]])
        option_names = list(options.keys()) + ["no_action"]
        best_idx = np.argmax(stacked, axis=1)
        best_value = stacked[np.arange(len(cp)), best_idx]

        total_value = best_value.sum()
        print(f"  Total expected net value this month: ${total_value:,.0f}", flush=True)
        print(f"  (n={len(cp):,} active policies scored)", flush=True)
        print("  Contact plan:", flush=True)
        for i, name in enumerate(option_names):
            n_i = int((best_idx == i).sum())
            val_i = best_value[best_idx == i].sum()
            print(f"    {name:22s} n={n_i:>8,} ({n_i/len(cp)*100:5.1f}%)  value=${val_i:>12,.0f}", flush=True)
        n_incentive = int(((best_idx == option_names.index("digital_discount")) |
                            (best_idx == option_names.index("phone_discount"))).sum())
        print(f"  Incentives offered: {n_incentive:,} ({n_incentive/len(cp)*100:.1f}% of book)", flush=True)

    # Where does expected value turn negative? (for baseline success rate)
    print(f"\n--- Break-even churn probability by channel (baseline scenario, no discount) ---", flush=True)
    for name, cost, rate in [("digital", DIGITAL_CONTACT_COST, BASELINE_SUCCESS_RATE),
                              ("phone", PHONE_CONTACT_COST, baseline_phone_rate)]:
        breakeven_p = cost / (rate * VALUE_RETAINED_FLAT)
        print(f"    {name:10s} (success rate {rate:.0%}): churn_probability must exceed {breakeven_p*100:.2f}% for a no-discount contact to be worth it", flush=True)

    # ==========================================================================
    # SECTION 2: Retrospective, type-differentiated analysis (val set has KNOWN
    # outcomes/types) -- answers "which types would benefit from an incentive?"
    # using type-specific success-rate assumptions grounded in the challenge
    # pack's own description of each type. NOT directly deployable live without
    # a type predictor (Q1/Q2 territory) -- shown as a decision-support/
    # further-research view, not the live monthly plan above.
    # ==========================================================================
    print("\n" + "=" * 78, flush=True)
    print("SECTION 2: Retrospective by-type analysis (known outcomes, val set)", flush=True)
    print("=" * 78, flush=True)
    print("(Type-specific success-rate assumptions, grounded in the challenge pack's own", flush=True)
    print(" description of each type -- NOT directly deployable live without a type", flush=True)
    print(" predictor; shown to answer 'which types would benefit from an incentive'.", flush=True)
    print(" Discount options use the same BASELINE_DISCOUNT_MULTIPLIER as Section 1.)", flush=True)

    pos_mask = y_val == 1
    for t in sorted(set(cancel_type[pos_mask].tolist())):
        t_mask = pos_mask & (cancel_type == t)
        n_t = int(t_mask.sum())
        mult = TYPE_SUCCESS_MULTIPLIER.get(t, 1.0)
        digital_eff_rate = BASELINE_SUCCESS_RATE * mult
        phone_eff_rate = digital_eff_rate * BASELINE_PHONE_MULTIPLIER
        cp_t = churn_prob[t_mask]
        ap_t = annual_premium[t_mask]
        md_t = ap_t * MAX_DISCOUNT_PCT_OF_ANNUAL
        nv_phone_disc = expected_net_value(cp_t, phone_eff_rate * BASELINE_DISCOUNT_MULTIPLIER, VALUE_RETAINED_FLAT, md_t, PHONE_CONTACT_COST)
        nv_phone_nodisc = expected_net_value(cp_t, phone_eff_rate, VALUE_RETAINED_FLAT, 0, PHONE_CONTACT_COST)
        nv_digital_disc = expected_net_value(cp_t, digital_eff_rate * BASELINE_DISCOUNT_MULTIPLIER, VALUE_RETAINED_FLAT, md_t, DIGITAL_CONTACT_COST)
        nv_digital_nodisc = expected_net_value(cp_t, digital_eff_rate, VALUE_RETAINED_FLAT, 0, DIGITAL_CONTACT_COST)
        best_mean = np.maximum.reduce([nv_phone_disc, nv_phone_nodisc, nv_digital_disc, nv_digital_nodisc, np.zeros_like(cp_t)]).mean()
        print(f"\n  {t} (n={n_t:,} actual cancellations in val, assumed success rate: digital {digital_eff_rate:.0%} / phone {phone_eff_rate:.0%}):", flush=True)
        print(f"    mean best expected net value per policy: ${best_mean:,.2f}", flush=True)
        print(f"    mean expected net value, phone+discount: ${nv_phone_disc.mean():,.2f}", flush=True)
        print(f"    mean expected net value, phone no discount: ${nv_phone_nodisc.mean():,.2f}", flush=True)
        print(f"    mean expected net value, digital+discount: ${nv_digital_disc.mean():,.2f}", flush=True)
        print(f"    mean expected net value, digital no discount: ${nv_digital_nodisc.mean():,.2f}", flush=True)

    # ==========================================================================
    # SECTION 3: Sensitivity -- premium-scaled value_retained as an alternative
    # to the flat $1,000 assumption (additional insight, not a replacement).
    # ==========================================================================
    print("\n" + "=" * 78, flush=True)
    print("SECTION 3: Sensitivity check -- premium-scaled value_retained (vs flat $1,000)", flush=True)
    print("=" * 78, flush=True)
    print("(Additional insight, not a replacement for HBF's flat $1,000 assumption --", flush=True)
    print(" shows how the plan would shift if higher-premium policies are assumed more", flush=True)
    print(" valuable to retain, scaled so the book average still equals $1,000.)", flush=True)
    scale_factor = VALUE_RETAINED_FLAT / annual_premium[month_mask].mean()
    value_retained_scaled = annual_premium[month_mask] * scale_factor
    nv_scaled = expected_net_value(cp, baseline_phone_rate, value_retained_scaled, 0, PHONE_CONTACT_COST)
    nv_flat = expected_net_value(cp, baseline_phone_rate, VALUE_RETAINED_FLAT, 0, PHONE_CONTACT_COST)
    n_flip = int(((nv_scaled > 0) != (nv_flat > 0)).sum())
    print(f"  Book average annual premium: ${annual_premium[month_mask].mean():,.0f} (scale factor applied: {scale_factor:.3f})", flush=True)
    print(f"  Policies whose phone-contact recommendation FLIPS (worth it vs not) under premium-scaled value: {n_flip:,}", flush=True)
    print(f"  Total expected net value, flat $1,000: ${nv_flat.sum():,.0f}", flush=True)
    print(f"  Total expected net value, premium-scaled: ${nv_scaled.sum():,.0f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, required=True)
    args = parser.parse_args()
    run_cost_benefit(args.window)
