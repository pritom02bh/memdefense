#!/bin/bash
# scripts/run_v3.sh -- unattended v3 regeneration (fourth batch, __v3d,
# extraction-prompt date pinned to 2026-10-30), then Stage 4c (rerun),
# the label gate and Stage 6.
#
#   nohup scripts/run_v3.sh > /dev/null 2>&1
#
# Runs `python -m runner.v3 --run` under caffeinate with
# MEMDEF_RESULTS=results_v3 and MEMDEF_PROMPT_DATE=2026-10-30, appending
# to results_v3/run.log.
#   exit 0      -> refresh the manifest (final ingestion times, prompt
#                  dates) and PROTOCOL.md, Stage 4c rerun (read-only) into
#                  results_v3/stage4c_report.txt (the previous one is kept as
#                  stage4c_report_v3c.txt), then scripts.stage6_gate --batch v3d:
#                  if it says go, Stage 6 into results_v3/stage6_final_report.txt
#                  with the per-cell table and the figure; if it says wait
#                  (exit 7), stop there (results_v3/STAGE6_WAITING.txt says why)
#   exit 4      -> retries exhausted: wait 15 minutes and resume, at most 3 times
#   exit 2/3/5  -> stop; the runner has written results_v3/STOPPED.txt
#   anything else, or a 4th exit 4 -> stop and write results_v3/STOPPED.txt
# Refuses to start while results_v3/STOPPED.txt exists. On the first
# start of the batch it snapshots the store fingerprints and the existing
# cell directories (read-only) for Stage 4c.

set -u
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"
LOG=results_v3/run.log
STOP=results_v3/STOPPED.txt
BATCH=v3d
export MEMDEF_RESULTS=results_v3
export MEMDEF_PROMPT_DATE=2026-10-30
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[wrapper $(stamp)] $*" >> "$LOG"; }
stop() { printf "%s  %s\n%s\n" "$(stamp)" "$1" "$2" > "$STOP"; say "stopped: $1"; }

if [ -e "$STOP" ]; then
  say "$STOP exists; not starting"
  exit 1
fi

if [ ! -e results_v3/store_fingerprint_before_$BATCH.json ] || [ ! -e results_v3/cells_before_$BATCH.sha256 ]; then
  say "snapshot before the $BATCH batch (store fingerprints, cell files)"
  if ! "$PY" -m scripts.stage4b_report --snapshot --batch $BATCH >> "$LOG" 2>&1; then
    stop "snapshot failed" "see run.log"
    exit 1
  fi
fi

resumes=0
while true; do
  say "starting runner.v3 --run (pid $$, resume $resumes/3)"
  caffeinate -i "$PY" -m runner.v3 --run >> "$LOG" 2>&1
  code=$?
  say "runner.v3 exited with code $code"
  echo "$code" > results_v3/run_exit_code
  if [ "$code" -eq 0 ]; then
    break
  fi
  if [ "$code" -eq 4 ] && [ "$resumes" -lt 3 ]; then
    resumes=$((resumes + 1))
    say "retries exhausted; resuming in 15 minutes ($resumes/3)"
    sleep 900
    continue
  fi
  case "$code" in
    2|3|5) say "stopped with code $code (runner wrote $STOP)" ;;
    4) stop "exit 4" "retries exhausted again after 3 resumes; see run.log" ;;
    *) stop "exit $code" "unexpected exit; see run.log" ;;
  esac
  exit "$code"
done

say "run complete; refreshing the manifest (final ingestion times) and PROTOCOL.md"
if ! "$PY" -m runner.v3 --manifest >> "$LOG" 2>&1 || ! "$PY" -m runner.v3 --protocol >> "$LOG" 2>&1; then
  stop "manifest/protocol refresh failed" "see run.log"
  exit 1
fi

say "Stage 4c report, rerun after __$BATCH (read-only)"
if [ -e results_v3/stage4c_report.txt ] && [ ! -e results_v3/stage4c_report_v3c.txt ]; then
  cp results_v3/stage4c_report.txt results_v3/stage4c_report_v3c.txt
fi
"$PY" -m scripts.stage4b_report --batch $BATCH > results_v3/stage4c_report.txt 2>&1
s4=$?
say "Stage 4c exited with code $s4"
if [ "$s4" -ne 0 ]; then
  stop "stage4c exit $s4" "Stage 4c checks failed; see stage4c_report.txt"
  exit "$s4"
fi

say "label gate (carry exact-text labels; new sheet; unsure drops)"
"$PY" -m scripts.stage6_gate --batch $BATCH >> "$LOG" 2>&1
gate=$?
say "label gate exited with code $gate"
if [ "$gate" -eq 7 ]; then
  say "Stage 6 not run: see results_v3/STAGE6_WAITING.txt"
  exit 7
elif [ "$gate" -ne 0 ]; then
  stop "label gate exit $gate" "see run.log"
  exit "$gate"
fi

say "Stage 6 (read-only) into results_v3/stage6_final_report.txt"
L=results_v3/stage6_final_logs
mkdir -p "$L"
"$PY" -m scripts.rescore_abstention > "$L/rescore_abstention.txt" 2>&1 \
  && "$PY" -m scripts.stage6_report --json stage6_final.json > results_v3/stage6_final_report.txt 2> "$L/stage6_stderr.txt" \
  && "$PY" -m scripts.stats --json > "$L/stats.txt" 2>&1 \
  && "$PY" -m scripts.make_percell_table > "$L/make_percell_table.txt" 2>&1 \
  && "$PY" fig.py > "$L/fig.txt" 2>&1
s6=$?
say "Stage 6 exited with code $s6"
if [ "$s6" -ne 0 ]; then
  stop "stage6 exit $s6" "see results_v3/stage6_final_logs/"
  exit "$s6"
fi
{
  echo
  echo "REGENERATED (read-only on the cells; logs in results_v3/stage6_final_logs/)"
  echo "  results_v3/adversarial_rescored.json, results_v3/stats.json, results_v3/stage6_final.json"
  echo "  results_v3/figures/tab_percell.tex, results_v3/figures/fig_cost_utility.pdf/.png"
  echo
  sed -n '/pooled check/,$p' "$L/make_percell_table.txt" | sed 's/^/  /'
} >> results_v3/stage6_final_report.txt
say "all done"
exit 0
