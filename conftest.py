"""pytest rootdir config — make `src` importable from tests."""
import os
import sys
from pathlib import Path

# XGBoost and PyTorch both load OpenMP; on macOS they deadlock / segfault
# when both runtimes coexist. Force single-threaded before any import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
