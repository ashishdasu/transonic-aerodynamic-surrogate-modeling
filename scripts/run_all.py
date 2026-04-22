"""Single entry point invoked by the Makefile.

    python scripts/run_all.py --stage {eda,train,eval,all}

Kept deliberately thin: each stage delegates to src/. No logic here that
could hide in a non-obvious place.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_eda() -> None:
    print("[eda] TODO Phase 2")


def run_train() -> None:
    print("[train] TODO Phase 3-6")


def run_eval() -> None:
    print("[eval] TODO Phase 7")


STAGES = {"eda": run_eda, "train": run_train, "eval": run_eval}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    args = ap.parse_args()

    if args.stage == "all":
        for s in ("eda", "train", "eval"):
            STAGES[s]()
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
