# tests/__init__.py
#
# Imported before any test module. Every path in core.paths resolves from
# MEMDEF_RESULTS at import time, so it is pointed at a throwaway root here,
# before any project import, whatever order the modules are loaded in.
# The root is named results_v3 so runner/v3.py's namespace guard applies
# exactly as in the real regeneration.
import os
import tempfile

TMP = tempfile.mkdtemp(prefix="memdef_test_")
os.environ["MEMDEF_RESULTS"] = os.path.join(TMP, "results_v3")
