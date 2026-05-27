"""Roda um único baseline em processo isolado (uso de background bash)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_train  # noqa: E402
from src.models import MODEL_REGISTRY  # noqa: E402
from src.preprocessing import (  # noqa: E402
    build_baseline_pipeline,
    build_robust_pipeline,
    build_smote_pipeline,
    build_target_encoding_pipeline,
)
from src.search import run_search  # noqa: E402
from src.utils import set_global_seed  # noqa: E402

FACTORIES = {
    "baseline": build_baseline_pipeline,
    "smote": build_smote_pipeline,
    "robust": build_robust_pipeline,
    "target_enc": build_target_encoding_pipeline,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--tag", default="baseline", choices=list(FACTORIES))
    args = ap.parse_args()

    set_global_seed()
    spec = MODEL_REGISTRY[args.model]
    factory = FACTORIES[args.tag]

    X, y = load_train()
    t0 = time.time()
    print(f">>> START {args.model} tag={args.tag}  ({spec.search}, grid={spec.param_grid})", flush=True)
    s = run_search(spec, X, y, factory, tag=args.tag)
    dt = time.time() - t0
    print(
        f">>> DONE  {args.model} tag={args.tag}  best_F1={s.best_score_:.4f}  "
        f"best_params={s.best_params_}  elapsed={dt:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
