#!/usr/bin/env python3
"""
Remove the one-time patch scripts and their backups.

    python3 cleanup.py           # show what would go, delete nothing
    python3 cleanup.py --delete  # actually delete

Refuses to delete anything unless the patches it is cleaning up are
confirmed applied. A backup is only junk once the thing it was insuring
against has succeeded -- deleting run.py.bak while run.py is unpatched
would throw away the only good copy.
"""
import pathlib
import sys

ROOT = pathlib.Path(".")

# (path, why it can go, file that must contain this marker for it to be spent)
SPENT = [
    ("patch_seed.py", "applied: runner/run.py has --seed",
     "runner/run.py", 'ap.add_argument("--seed"'),
    ("patch_stats_seeds.py", "applied: stats.py collapses replicates",
     "scripts/stats.py", "def select_replicates("),
    ("patch_stats_rescored.py", "applied: stats.py reads corrected labels",
     "scripts/stats.py", "RESCORED = pathlib.Path"),
    ("runner/run.py.bak", "pre-seed backup",
     "runner/run.py", 'ap.add_argument("--seed"'),
    ("scripts/stats.py.bak", "pre-replicate backup",
     "scripts/stats.py", "def select_replicates("),
    ("scripts/stats.py.bak2", "pre-rescore backup",
     "scripts/stats.py", "RESCORED = pathlib.Path"),
]

# junk that accumulates but is regenerated on demand
GENERATED = [
    "paper/main.aux", "paper/main.log", "paper/main.out",
    "paper/main.bbl", "paper/main.blg", "paper/main.toc",
]


def main():
    do_it = "--delete" in sys.argv
    go, skip = [], []

    for path, why, guard_file, marker in SPENT:
        p = ROOT / path
        if not p.exists():
            continue
        guard = ROOT / guard_file
        if not guard.exists():
            skip.append((path, f"KEEP -- {guard_file} missing, cannot verify"))
        elif marker not in guard.read_text():
            skip.append((path, f"KEEP -- {guard_file} not patched; this may be your only good copy"))
        else:
            go.append((path, why))

    for path in GENERATED:
        p = ROOT / path
        if p.exists():
            go.append((path, "LaTeX build artifact, regenerated on compile"))

    if skip:
        print("NOT deleting:")
        for path, why in skip:
            print(f"  {path:32s} {why}")
        print()

    if not go:
        print("Nothing to clean up.")
        return 0

    print(f"{'Would delete' if not do_it else 'Deleting'}:")
    for path, why in go:
        print(f"  {path:32s} {why}")

    if not do_it:
        print(f"\n{len(go)} file(s). Re-run with --delete to remove them.")
        return 0

    for path, _ in go:
        (ROOT / path).unlink()
    print(f"\nRemoved {len(go)} file(s).")
    print("Nothing in results/ was touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())