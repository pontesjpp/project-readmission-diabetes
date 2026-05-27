"""Roda uma variante de pré-processamento para uma lista de modelos.

Uso:
    uv run python scripts/run_variant_sweep.py --tag smote --models knn,svm,lightgbm

Se `--models` for omitido, percorre todos os modelos do MODEL_REGISTRY (exceto
incompatibilidades documentadas — SMOTE × Stacking).
"""

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
    build_undersampling_pipeline,
)
from src.search import get_or_run_search  # noqa: E402
from src.utils import set_global_seed  # noqa: E402

FACTORIES = {
    "baseline": build_baseline_pipeline,
    "smote": build_smote_pipeline,
    "robust": build_robust_pipeline,
    "target_enc": build_target_encoding_pipeline,
    "undersample": build_undersampling_pipeline,
}

INCOMPATIBLE = {
    ("smote", "stacking"),        # StackingClassifier não casa com imblearn pipeline.
    ("undersample", "stacking"),  # mesma razão.
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(FACTORIES))
    ap.add_argument("--models", default="", help="csv de nomes; vazio = todos")
    args = ap.parse_args()

    set_global_seed()
    factory = FACTORIES[args.tag]
    if args.models.strip():
        names = [n.strip() for n in args.models.split(",") if n.strip()]
    else:
        names = list(MODEL_REGISTRY)

    X, y = load_train()
    t_start = time.time()
    for name in names:
        spec = MODEL_REGISTRY[name]
        if (args.tag, name) in INCOMPATIBLE:
            print(f">>> SKIP   {name} | {args.tag} (incompat documentada)", flush=True)
            continue
        t0 = time.time()
        print(f">>> START  {name} | {args.tag}  (grid_size={len(spec.param_grid)})", flush=True)
        try:
            s = get_or_run_search(spec, X, y, factory, tag=args.tag)
            print(
                f">>> DONE   {name} | {args.tag}  F1={s.best_score_:.4f}  "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
        except Exception as e:
            print(f">>> ERROR  {name} | {args.tag}  -> {type(e).__name__}: {e}", flush=True)

    print(f"\n[{args.tag}] total elapsed = {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
