# scripts/probe_infer_mode.py
#
# QUESTION THIS ANSWERS: is the "419 turns -> 20 memories" ceiling caused
# by Mem0's extraction LLM (infer=True, the default), or by something in
# our calling pattern?
#
# METHOD: ingest the SAME 100 turns into two separate namespaces:
#   probe_infer_on   -- add(..., infer=True)   extraction LLM decides
#   probe_infer_off  -- add(..., infer=False)  store raw, no LLM
# Then count what each namespace actually holds.
#
# READING THE RESULT:
#   infer_off ~= 100, infer_on ~= 20  -> ceiling is the extractor's doing.
#       Consolidation is a real framework property. Report with confidence.
#   both ~= 20                        -> ceiling is NOT the extractor;
#       something in our adapter/config truncates. Do not report the
#       consolidation claim until found.
#   infer_off fails                   -> mem0 2.0.15 changed the API;
#       paste the error.
#
#   python3 -m scripts.probe_infer_mode
#
# Cost: infer_off is nearly free (embeddings only). infer_on re-runs the
# extractor on 100 turns (~3 min, same as before).

import dataclasses
import sys
import time

from adapters.mem0_adapter import Mem0Adapter
from core.dataset import load_locomo

TURNS = 100


def count_memories(adapter: Mem0Adapter, namespace: str) -> int:
    adapter._namespace = namespace
    return len(adapter.dump_memories())


def ingest(adapter: Mem0Adapter, conv, namespace: str, infer: bool) -> tuple[int, float]:
    adapter.reset(namespace)
    adapter._namespace = namespace
    errors = 0
    t0 = time.monotonic()
    for i, turn in enumerate(conv.turns, 1):
        try:
            adapter._mem.add(
                [{"role": "user",
                  "content": f"[Conversation date: {turn.timestamp}] {turn.speaker}: {turn.text}"}],
                user_id=namespace,
                metadata={"turn_id": turn.turn_id, "session_id": turn.session_id},
                infer=infer,
            )
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    turn {i} error: {type(e).__name__}: {e}", flush=True)
        if i % 25 == 0:
            print(f"    {i}/{TURNS}", flush=True)
    return errors, time.monotonic() - t0


def main():
    convs = load_locomo("data/locomo/locomo10.json")
    conv = dataclasses.replace(convs[0], turns=convs[0].turns[:TURNS])
    adapter = Mem0Adapter(run_id="probe", cell_id="probe_infer_mode")

    print(f"Probing with the same {TURNS} turns of {convs[0].conv_id}.\n")

    print("1/2  infer=False (raw storage, no extraction LLM)...")
    errs_off, secs_off = ingest(adapter, conv, "probe_infer_off", infer=False)
    n_off = count_memories(adapter, "probe_infer_off")
    print(f"     stored: {n_off} memories, {errs_off} errors, {secs_off/60:.1f} min\n")

    print("2/2  infer=True (default extraction pipeline)...")
    errs_on, secs_on = ingest(adapter, conv, "probe_infer_on", infer=True)
    n_on = count_memories(adapter, "probe_infer_on")
    print(f"     stored: {n_on} memories, {errs_on} errors, {secs_on/60:.1f} min\n")

    print("=" * 56)
    print(f"  infer=False : {n_off:4d} memories from {TURNS} turns")
    print(f"  infer=True  : {n_on:4d} memories from {TURNS} turns")
    print("=" * 56)

    if n_off >= TURNS * 0.8 and n_on < TURNS * 0.4:
        print("\nVERDICT: the ceiling is the extraction pipeline. Raw storage")
        print("keeps everything; the extractor consolidates aggressively.")
        print("Consolidation is a real framework property -- reportable.")
    elif n_off < TURNS * 0.4:
        print("\nVERDICT: raw mode ALSO capped. The ceiling is not the")
        print("extractor -- something in the adapter, vector store config,")
        print("or dump_memories() truncates. Investigate before making any")
        print("consolidation claim.")
    else:
        print("\nVERDICT: intermediate result -- paste the numbers back for")
        print("interpretation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())