"""
Shared settings for the Churn Busters pipeline (HBF Challenge 1: predicting policy cancellations).

Every script imports this file, so paths, months, features, model settings and the
cost-benefit assumptions are defined once.

Folders
  HBF_DATA  the folder holding HBF's four data extracts (DATA_Monthly_Active_Policy_Spline, ...)
  HBF_WORK  large intermediate files (feature tables, monthly arrays); not part of the results
  ROOT      this submission folder: results/, models/ and live_demo/ are written here
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.abspath(os.environ.get("HBF_DATA", os.path.join(ROOT, "data")))
WORK = os.path.abspath(os.environ.get("HBF_WORK", os.path.join(ROOT, "work")))
CACHE = os.path.join(WORK, "monthly_arrays")

RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
LIVE_DEMO = os.path.join(ROOT, "live_demo")
TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# ---- HBF data extracts ------------------------------------------------------------------
SPINE = os.path.join(DATA, "DATA_Monthly_Active_Policy_Spline",
                     "HBF_2026_WADSIH_CHALLENGE-DATA-Monthly_Active_Policy_Spine.parquet")
CANCELLATIONS = os.path.join(DATA, "DATA_Cancellations", "Cancellations.parquet")
CLAIMS = os.path.join(DATA, "DATA_Claims", "Claims.parquet")
SURVEY = os.path.join(DATA, "DATA_Channel Survey Feedback", "Channel_Survey_Feedback.parquet")

# ---- Intermediate tables (in WORK) -----------------------------------------------------
LABELS = os.path.join(WORK, "labels.parquet")
FEATURES_CLAIMS = os.path.join(WORK, "features_claims.parquet")
FEATURES_SURVEY = os.path.join(WORK, "features_survey.parquet")
FEATURES_SPINE = os.path.join(WORK, "features_spine.parquet")
FEATURES = os.path.join(WORK, "features.parquet")
MODEL_TABLE = os.path.join(WORK, "model_table.parquet")
CATEGORY_CODES = os.path.join(MODELS, "category_codes.json")

# ---- Target and month split ------------------------------------------------------------
HORIZON_MONTHS = 3            # predict "will this policy cancel within the next 3 months?"
TRAIN_SHARE, VAL_SHARE = 0.70, 0.15   # of the labelled months, in date order; the rest is the locked test
GAP_MONTHS = 1                # one unused month after training and after validation

# ---- Features ---------------------------------------------------------------------------
NUMERIC = [
    "ACTIVE_ADULT_COUNT", "ACTIVE_CHILD_COUNT", "INCOME_TIER", "MONTHLY_PREMIUM", "LHC_LOADING_RATE",
    "TOTAL_CLAIM_COUNT_ROLL3M", "TOTAL_BENEFIT_PAID_ROLL3M", "TOTAL_OUT_OF_POCKET_ROLL3M",
    "HOSPITAL_CLAIM_COUNT_ROLL3M", "REVERSAL_FLAG_ROLL3M", "NO_BENEFIT_PAID_FLAG_ROLL3M",
    "TOTAL_CLAIM_COUNT_ROLL6M", "TOTAL_BENEFIT_PAID_ROLL6M", "TOTAL_OUT_OF_POCKET_ROLL6M",
    "HOSPITAL_CLAIM_COUNT_ROLL6M", "REVERSAL_FLAG_ROLL6M", "NO_BENEFIT_PAID_FLAG_ROLL6M",
    "TOTAL_CLAIM_COUNT_ROLL12M", "TOTAL_BENEFIT_PAID_ROLL12M", "TOTAL_OUT_OF_POCKET_ROLL12M",
    "HOSPITAL_CLAIM_COUNT_ROLL12M", "REVERSAL_FLAG_ROLL12M", "NO_BENEFIT_PAID_FLAG_ROLL12M",
    "MONTHS_SINCE_LAST_CLAIM", "MONTHS_SINCE_LAST_SURVEY", "LAST_OVERALL_EXPERIENCE_RATING",
    "LAST_NPS_RATING", "LAST_SERVICE_RATING", "RECENT_LOW_RATING_FLAG", "HAS_EVER_SURVEYED",
    "HAD_COMPLAINT_FLAG", "ARREARS_FLAG_LAG2M", "PREMIUM_CHANGE_3M", "NEGATIVE_SENTIMENT_CALLS_LAST_6M",
]
CATEGORICAL = [
    "POLICY_OWNER_AGE_BAND", "SCALE", "STATE_CODE", "CHANNEL_ACQUISITION", "TENURE_BAND",
    "YOUNGEST_DEPENDANT_AGE_BAND", "PRODUCT_HOLDINGS", "HOSPITAL_TIER", "EXTRAS_TIER",
    "HOSPITAL_EXCESS_AMT", "PAYMENT_METHOD_DESC", "PAYMENT_FREQUENCY_DESC", "ARREARS_FLAG",
    "MARKETING_OPTIN", "CALL_NEGATIVE_SENTIMENT", "RESOLUTION_TIME",
]
ALL_FEATURES = NUMERIC + CATEGORICAL

# Left out of every model on purpose: each is filled in or changes as people leave, so it would
# report the outcome rather than predict it (evidence in results/excluded_fields.json):
#   MARKETING_OPTIN  never changes; "Unknown" almost always ends in a cancellation
#   INCOME_TIER      blanked while a cancellation is being processed
#   ARREARS_FLAG     this month's arrears already triggers HBF's own process (the 2-month-old flag is kept)
EXCLUDED = ["MARKETING_OPTIN", "INCOME_TIER", "ARREARS_FLAG"]
MODEL_FEATURES = [f for f in ALL_FEATURES if f not in EXCLUDED]

LABEL_TEXT = {
    "ACTIVE_ADULT_COUNT": "Number of adults on the policy", "ACTIVE_CHILD_COUNT": "Number of children on the policy",
    "INCOME_TIER": "Income tier", "MONTHLY_PREMIUM": "Monthly premium",
    "LHC_LOADING_RATE": "Loyalty loading (joined later in life)",
    "TOTAL_CLAIM_COUNT_ROLL3M": "How many claims they've made (3mo)", "TOTAL_BENEFIT_PAID_ROLL3M": "Benefits paid out recently (3mo)",
    "TOTAL_OUT_OF_POCKET_ROLL3M": "Out-of-pocket costs recently (3mo)", "HOSPITAL_CLAIM_COUNT_ROLL3M": "Hospital claims recently (3mo)",
    "REVERSAL_FLAG_ROLL3M": "Claim reversals (3mo)", "NO_BENEFIT_PAID_FLAG_ROLL3M": "Claims with no benefit paid (3mo)",
    "TOTAL_CLAIM_COUNT_ROLL6M": "How many claims they've made (6mo)", "TOTAL_BENEFIT_PAID_ROLL6M": "Benefits paid out recently (6mo)",
    "TOTAL_OUT_OF_POCKET_ROLL6M": "Out-of-pocket costs recently (6mo)", "HOSPITAL_CLAIM_COUNT_ROLL6M": "Hospital claims recently (6mo)",
    "REVERSAL_FLAG_ROLL6M": "Claim reversals (6mo)", "NO_BENEFIT_PAID_FLAG_ROLL6M": "Claims with no benefit paid (6mo)",
    "TOTAL_CLAIM_COUNT_ROLL12M": "How many claims they've made (12mo)", "TOTAL_BENEFIT_PAID_ROLL12M": "Benefits paid out recently (12mo)",
    "TOTAL_OUT_OF_POCKET_ROLL12M": "Out-of-pocket costs recently (12mo)", "HOSPITAL_CLAIM_COUNT_ROLL12M": "Hospital claims recently (12mo)",
    "REVERSAL_FLAG_ROLL12M": "Claim reversals (12mo)", "NO_BENEFIT_PAID_FLAG_ROLL12M": "Claims with no benefit paid (12mo)",
    "MONTHS_SINCE_LAST_CLAIM": "Time since their last claim", "MONTHS_SINCE_LAST_SURVEY": "Time since their last survey response",
    "LAST_OVERALL_EXPERIENCE_RATING": "Their last satisfaction rating", "LAST_NPS_RATING": "Their last NPS rating",
    "LAST_SERVICE_RATING": "Their last service rating", "RECENT_LOW_RATING_FLAG": "Recent low satisfaction rating",
    "HAS_EVER_SURVEYED": "Whether they've ever completed a survey", "HAD_COMPLAINT_FLAG": "Recent complaint on file",
    "ARREARS_FLAG_LAG2M": "Payment arrears history", "PREMIUM_CHANGE_3M": "Recent premium change",
    "NEGATIVE_SENTIMENT_CALLS_LAST_6M": "Negative-sentiment calls, last 6mo", "POLICY_OWNER_AGE_BAND": "Age",
    "SCALE": "Membership type (single/couple/family)", "STATE_CODE": "State", "CHANNEL_ACQUISITION": "How they joined",
    "TENURE_BAND": "How long they've been a member", "YOUNGEST_DEPENDANT_AGE_BAND": "Youngest dependant's age",
    "PRODUCT_HOLDINGS": "What cover they hold", "HOSPITAL_TIER": "Hospital cover tier", "EXTRAS_TIER": "Extras cover tier",
    "HOSPITAL_EXCESS_AMT": "Hospital excess amount", "PAYMENT_METHOD_DESC": "How they pay",
    "PAYMENT_FREQUENCY_DESC": "How often they pay", "ARREARS_FLAG": "Currently in arrears",
    "MARKETING_OPTIN": "Marketing sign-up status", "CALL_NEGATIVE_SENTIMENT": "Recent call sentiment",
    "RESOLUTION_TIME": "How long issues take to resolve",
}

# ---- Cancellation types -----------------------------------------------------------------
TYPES = ["Switcher", "Leave PHI", "System Lapsed", "Non-Recoverable"]
UNCLASSIFIED = "(unclassified)"
ALL_TYPES = TYPES + [UNCLASSIFIED]

# ---- Model settings (LightGBM) ----------------------------------------------------------
ROUNDS = 150                  # fixed number of trees; the best round is chosen on validation PR-AUC
SEED = 42


def lgb_params(scale_pos_weight=1.0, threads=None):
    return {"objective": "binary", "metric": ["auc", "average_precision"], "scale_pos_weight": scale_pos_weight,
            "num_leaves": 31, "max_bin": 127, "learning_rate": 0.08, "feature_fraction": 0.7,
            "bagging_fraction": 0.6, "bagging_freq": 1, "min_data_in_leaf": 100,
            "num_threads": threads or int(os.environ.get("HBF_THREADS", 4)), "verbose": -1, "seed": SEED,
            "deterministic": True, "force_col_wise": True}


# ---- Retention economics (HBF challenge pack, slide 7) ----------------------------------
VALUE_RETAINED = 1000.0       # an average policy saved is worth $1,000
PHONE_COST = 20.0             # a phone call costs $20
DIGITAL_COST = 0.10           # a digital message costs 10 cents
MAX_DISCOUNT = 0.12           # discount of up to 12% of annual premium, which reduces the value saved

# Our assumptions (HBF did not give a success rate); each is tested over a range
SUCCESS_RATES = [0.10, 0.20, 0.30]   # chance a digital message saves a policy that would have left
BASE_SUCCESS = 0.20
PHONE_LIFT = 1.5              # a call is 1.5x as persuasive as a digital message
DISCOUNT_LIFT = 1.3           # a discount makes an offer 1.3x as persuasive
TYPE_SUCCESS_LIFT = {         # from the challenge pack's description of each type (slide 6)
    "Switcher": 1.5,          # "highly contestable with the right, timely offer"
    "Leave PHI": 1.0,         # "value/affordability levers matter most"
    "System Lapsed": 0.75,    # "often affordability/disengagement related"
    "Non-Recoverable": 0.1,   # "can't realistically be saved"
    UNCLASSIFIED: 0.5,
}
