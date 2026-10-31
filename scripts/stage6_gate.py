# scripts/stage6_gate.py
#
# After a batch (scripts/run_v3.sh runs it after Stage 4c). No API
# calls. Decides whether Stage 6 can run unattended:
#   1. adversarial labels for the batch's answers (--batch v3c | v3d,
#      default v3d): exact-text carry-over, and a new sheet for the regex
#      failures still unlabeled (scripts/make_label_sheets.py --<batch>);
#   2. the drop audit must have no unsure and no unresolved item left.
# Exit 0: go. Exit 7: wait -- the reasons are printed and written to
# <root>/STAGE6_WAITING.txt.
#
#   MEMDEF_RESULTS=results_v3 python3 -m scripts.stage6_gate [--batch v3d]

from __future__ import annotations

import datetime
import sys

from core import labels as lb
from core.paths import RESULTS
from scripts import make_label_sheets as ms

WAITING = RESULTS / "STAGE6_WAITING.txt"


def main() -> int:
    batch = sys.argv[sys.argv.index("--batch") + 1] if "--batch" in sys.argv else "v3d"
    if batch not in ms.BATCH_NAMES:
        sys.exit(f"--batch must be one of {sorted(ms.BATCH_NAMES)}")
    sys.argv = [sys.argv[0], "--" + batch]
    reasons = []
    if ms.main() == 7:
        reasons.append(f"new adversarial sheet to label: {ms.LABELING / f'adversarial_labels_{batch}.txt'} "
                       f"(key in {ms.KEYS}, private)")
    f = lb.final(lb.DROP, strict=False)
    unsure = sorted(k for k, v in f.labels.items() if lb.is_unsure(lb.DROP, v))
    print(f"drop audit: {len(f.labels)} final labels, {len(unsure)} unsure {unsure}, "
          f"{len(f.unresolved)} unresolved")
    if unsure or f.unresolved:
        reasons.append(f"drop audit: {len(unsure)} unsure {unsure}, {len(f.unresolved)} unresolved "
                       f"(fill {lb.DROP.dir / 'disagreements.txt'}, then "
                       f"python3 -m scripts.final_labels --resolve)")
    if reasons:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        WAITING.write_text(f"{now}  Stage 6 not run\n" + "\n".join(f"- {r}" for r in reasons) + "\n")
        print("WAIT: " + "; ".join(reasons))
        return 7
    if WAITING.exists():
        WAITING.unlink()
    print("GO: no new sheet, no unsure drops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
