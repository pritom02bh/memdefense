# core/manifest.py
#
# The cell manifest: <root>/final_cells.json, written by
# `python3 -m runner.v3 --manifest`. It names the cells the paper uses:
#
#   role "main"   one cell per condition (none ... stacked), conversation
#                 and replicate; within a conversation every main cell
#                 answers from memory extracted in the same period
#                 (store_period "Aug" or "Oct").
#   role "split"  the model-swap split: the 15 old (v2) none and 15 old
#                 rerank cells from results/, and the 15 baseline-mini
#                 cells -- all on August memory.
#
# Superseded cells stay on disk; they are excluded only by not being
# listed. Analysis scripts take their cells from here (cell_dirs()).
# A root without a manifest (the original results/) keeps the old
# behaviour: every cell directory under it.

from __future__ import annotations

import json
import pathlib

from core.paths import CELLS, REPO_ROOT, RESULTS

MANIFEST_NAME = "final_cells.json"
MANIFEST = RESULTS / MANIFEST_NAME
ROLES = ("main", "split")


def load(path: pathlib.Path = MANIFEST) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def root_dir(name: str) -> pathlib.Path:
    """A manifest root by name: this root, or a sibling under the repo
    (the old results/)."""
    return RESULTS if name == RESULTS.name else REPO_ROOT / name


def entry_dir(e: dict) -> pathlib.Path:
    return root_dir(e["root"]) / "cells" / e["cell_id"]


def entries(roles=ROLES, local_only: bool = False, manifest: dict | None = None) -> list[dict]:
    """Manifest entries in the given roles; local_only keeps only the
    cells that live under this root (not the old results/ cells)."""
    m = manifest or load()
    if m is None:
        raise SystemExit(f"no manifest at {MANIFEST} (python3 -m runner.v3 --manifest)")
    return [e for e in m["cells"] if e["role"] in roles
            and (not local_only or e["root"] == RESULTS.name)]


def required() -> bool:
    """Every root except the original results/ must have a manifest."""
    return RESULTS != REPO_ROOT / "results"


def cell_dirs(roles=ROLES, local_only: bool = False) -> list[pathlib.Path]:
    """The cell directories an analysis script should read: the
    manifest's, or -- for a root without one -- every directory under
    <root>/cells."""
    if MANIFEST.exists():
        return [entry_dir(e) for e in entries(roles, local_only)]
    if required():
        raise SystemExit(f"no manifest at {MANIFEST} (python3 -m runner.v3 --manifest)")
    return sorted(d for d in CELLS.iterdir() if d.is_dir()) if CELLS.exists() else []


def entry_for(cell_id: str) -> dict | None:
    m = load()
    return next((e for e in m["cells"] if e["cell_id"] == cell_id), None) if m else None


def baseline_dir(cell_id: str) -> pathlib.Path:
    """The main none cell paired with a main cell (same conversation and
    replicate); without a manifest, the none cell named like it."""
    e = entry_for(cell_id)
    if e is None:
        parts = cell_id.split("__")
        parts[2] = "none"
        return CELLS / "__".join(parts)
    m = load()
    base = next(x for x in m["cells"] if x["role"] == e["role"] and x["condition"] == "none"
                and x["conv_idx"] == e["conv_idx"] and x["seed"] == e["seed"])
    return entry_dir(base)
