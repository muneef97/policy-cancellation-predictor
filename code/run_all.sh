#!/usr/bin/env bash
# =============================================================================
# Churn Busters - HBF Challenge 1: run the whole pipeline, from HBF's data to the models, results and live demo.
#
#   ./run_all.sh --data /path/to/hbf/data          run every step
#   ./run_all.sh --data /path/to/hbf/data --from 11  re-run from step 11 (e.g. after changing an assumption)
#   ./run_all.sh --list                             show the steps
#
# The data folder must contain HBF's four extracts:
#   DATA_Monthly_Active_Policy_Spline/  DATA_Cancellations/  DATA_Claims/  DATA_Channel Survey Feedback/
# Large intermediate files go to ../work (override with --work). Every run is logged to ../logs/.
# Typical time on a laptop: 30 to 60 minutes. Needs about 10 GB of free disk space.
# =============================================================================
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$CODE/.." && pwd)"
PY="${PYTHON:-python3}"
FROM=1; TO=13; LIST=0
export HBF_DATA="${HBF_DATA:-$ROOT/data}" HBF_WORK="${HBF_WORK:-$ROOT/work}" PYTHONWARNINGS=ignore

STEPS="
1|01_build_labels.py|Target: cancels within 3 months; months ahead, type and reason; month split
2|02_build_features.py|Features: claims history, survey answers, policy details and trends
3|03_prepare_model_data.py|Join labels and features; fixed category codes; one array file per month
4|04_train_models.py|Final model and the exact-month models (LightGBM)
5|05_final_check.py|One-off check on the locked test months (Mar-May 2026)
6|06_group_check.py|Does it work equally well for every state, age, tenure, channel and income tier?
7|07_type_models.py|Question 1: what predicts each cancellation type
8|08_granularity.py|Question 2: can the type, and the detailed reason, be told apart?
9|09_how_early.py|Question 3: how early each type shows up
10|10_risk_journeys.py|Question 3: how risk built up before leaving (four groups of leavers)
11|11_retention_plan.py|Question 4: calibration and the best action for every policy
12|12_summary_data.py|Collect the summary numbers the live demo shows
13|13_build_live_demo.py|Build the live demo (HBF Retention Hub)
"

while [ $# -gt 0 ]; do
  case "$1" in
    --data) HBF_DATA="$2"; shift 2 ;;
    --work) HBF_WORK="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --list) LIST=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)"; exit 1 ;;
  esac
done
if [ "$LIST" = 1 ]; then echo "$STEPS" | sed '/^$/d' | awk -F'|' '{printf "  %2s  %-26s %s\n", $1, $2, $3}'; exit 0; fi
export HBF_DATA="$(cd "$HBF_DATA" 2>/dev/null && pwd || echo "$HBF_DATA")" HBF_WORK

mkdir -p "$ROOT/logs" "$HBF_WORK"
LOG="$ROOT/logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "Churn Busters pipeline - steps $FROM to $TO"
echo "  data: $HBF_DATA"
echo "  work: $HBF_WORK"
echo "  log:  $LOG"
for f in "DATA_Monthly_Active_Policy_Spline" "DATA_Cancellations" "DATA_Claims" "DATA_Channel Survey Feedback"; do
  [ -d "$HBF_DATA/$f" ] || { echo "Missing $HBF_DATA/$f - point --data at HBF's data folder"; exit 1; }
done
"$PY" - <<'PYCHECK'
import importlib.util, sys
need = ["pandas", "numpy", "pyarrow", "duckdb", "lightgbm", "sklearn"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("Missing Python packages: " + ", ".join(missing) + "\n  install with: pip install -r requirements.txt")
PYCHECK

cd "$CODE"
START=$SECONDS
echo "$STEPS" | sed '/^$/d' | while IFS='|' read -r n script what; do
  if [ "$n" -ge "$FROM" ] && [ "$n" -le "$TO" ]; then
    echo; echo "== Step $n: $what"
    T=$SECONDS
    "$PY" "$script"
    echo "   ($((SECONDS - T))s)"
  fi
done
echo; echo "Done in $(( (SECONDS - START) / 60 )) min. Results: $ROOT/results  Models: $ROOT/models  Live demo: $ROOT/live_demo"
