# scripts/fix_judge_cache_key.py
#
# One-time patch. Run once, then delete.
#
# PROBLEM: core/judge.py caches grades keyed on
# sha256(question|gold|predicted). The judge PROMPT is not part of that
# key. So if prompts/judge.txt changes, every previously cached item
# returns its OLD grade with no error and no warning -- the new prompt
# is silently ignored for anything already seen.
#
# We already hit this once during calibration (had to rm the cache by
# hand). Mid-experiment it would be worse: some cells graded under the
# old prompt, some under the new, no way to tell which from the results.
#
# FIX: include a hash of the prompt file in the cache key. A prompt edit
# then produces different keys, so old entries are simply never matched
# and fresh grades are computed automatically. Old entries stay on disk
# harmlessly as a record of what the previous prompt decided.

import pathlib
import sys

JUDGE = pathlib.Path("core/judge.py")


def main():
    if not JUDGE.exists():
        print(f"Cannot find {JUDGE}. Run from the project root.")
        return 1

    s = JUDGE.read_text()

    if "_PROMPT_HASH" in s:
        print("Already patched -- cache key includes the prompt hash. Nothing to do.")
        return 0

    # 1. Compute a prompt hash once at import, right after the template loads.
    old_load = '_prompt_template = PROMPT_PATH.read_text()'
    new_load = (
        '_prompt_template = PROMPT_PATH.read_text()\n'
        '\n'
        '# Hash of the prompt itself. Included in every cache key so that\n'
        '# editing prompts/judge.txt invalidates cached grades automatically\n'
        '# instead of silently returning judgments made under the old prompt.\n'
        '_PROMPT_HASH = hashlib.sha256(_prompt_template.encode("utf-8")).hexdigest()[:16]'
    )
    if old_load not in s:
        print("Could not find the prompt-loading line. Patch aborted, file unchanged.")
        return 1
    s = s.replace(old_load, new_load, 1)

    # 2. Fold the prompt hash into the cache key.
    old_key = 'raw = f"{question}|{gold}|{predicted}".encode("utf-8")'
    new_key = 'raw = f"{_PROMPT_HASH}|{question}|{gold}|{predicted}".encode("utf-8")'
    if old_key not in s:
        print("Could not find the cache-key line. Patch aborted, file unchanged.")
        return 1
    s = s.replace(old_key, new_key, 1)

    # 3. Record which prompt version produced each cached grade, so the
    #    cache file is self-describing if you ever need to audit it.
    old_write = '''            "correct": result.correct,
            "reason": result.reason,'''
    new_write = '''            "correct": result.correct,
            "reason": result.reason,
            "prompt_hash": _PROMPT_HASH,'''
    if old_write in s:
        s = s.replace(old_write, new_write, 1)
    else:
        print("  (note: could not add prompt_hash to cache rows; key fix still applied)")

    JUDGE.write_text(s)
    print("Patched core/judge.py:")
    print("  - prompt hash computed at import")
    print("  - cache key now includes it")
    print("  - cached rows record which prompt graded them")
    print("\nNo need to clear the cache after prompt edits from now on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())