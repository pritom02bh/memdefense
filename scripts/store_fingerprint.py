# scripts/store_fingerprint.py
#
# READ-ONLY fingerprint of every namespace in a local mem0 Qdrant store:
# point count and a sha256 over the sorted (point id, payload) pairs.
# Taken before the v3 run and again in Stage 4, it shows whether
# answering changed any store -- counts alone would miss an edit.
#
#   MEMDEF_RESULTS=results_v3 python3 -m scripts.store_fingerprint --out results_v3/store_fingerprint_before.json
#
# Opening a local Qdrant folder takes its lock: run it only while no
# other process has the store open, and never on results/mem0_qdrant.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from collections import defaultdict

from core.paths import QDRANT, REPO_ROOT


def fingerprint(store_path=QDRANT, collection: str = "memdefense") -> dict:
    if store_path.resolve() == (REPO_ROOT / "results" / "mem0_qdrant").resolve():
        raise PermissionError("refusing to open results/mem0_qdrant (opening it writes its lock file)")
    warnings.filterwarnings("ignore")
    from qdrant_client import QdrantClient
    client = QdrantClient(path=str(store_path))
    try:
        items = defaultdict(list)
        offset = None
        while True:
            pts, offset = client.scroll(collection, limit=5000, offset=offset,
                                        with_payload=True, with_vectors=False)
            for p in pts:
                items[(p.payload or {}).get("user_id")].append(
                    (str(p.id), json.dumps(p.payload, sort_keys=True)))
            if offset is None:
                break
    finally:
        client.close()
    out = {}
    for ns, rows in items.items():
        h = hashlib.sha256()
        for pid, payload in sorted(rows):
            h.update(pid.encode())
            h.update(payload.encode())
        out[ns] = {"points": len(rows), "sha256": h.hexdigest()}
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fp = fingerprint()
    with open(args.out, "w") as f:
        json.dump({"store": str(QDRANT), "namespaces": fp}, f, indent=2)
    print(f"wrote {args.out}: {len(fp)} namespaces, {sum(v['points'] for v in fp.values())} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
