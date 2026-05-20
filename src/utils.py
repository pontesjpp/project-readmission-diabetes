"""Seeds, paths e folds compartilhados entre experimentos.

Centraliza tudo que precisa ser idêntico entre runs para garantir comparações
pareadas (mesmos folds, mesmo random_state) e reprodutibilidade.
"""

from __future__ import annotations

import os
import pickle
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
N_FOLDS = 5

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
CV_RESULTS = REPORTS / "cv_results"


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Fixa seeds em python/numpy/PYTHONHASHSEED."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_cv(n_splits: int = N_FOLDS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Retorna o splitter usado em TODOS os experimentos.

    Reutilizar a mesma instância garante que comparações pareadas
    (Wilcoxon / paired t-test) sejam válidas.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, FIGURES, CV_RESULTS):
        p.mkdir(parents=True, exist_ok=True)


LABEL_ENCODER_PATH = DATA_PROCESSED / "label_encoder.pkl"


def save_label_encoder(le: LabelEncoder) -> None:
    """Persiste o LabelEncoder em disco para uso posterior."""
    ensure_dirs()
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)


def load_label_encoder() -> LabelEncoder:
    """Carrega o LabelEncoder persistido, ou None se não existe."""
    if LABEL_ENCODER_PATH.exists():
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            return pickle.load(f)
    return None


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Detecta se há GPU CUDA utilizável pelos modelos (hoje, XGBoost).

    A checagem casa o build do XGBoost (`USE_CUDA`) com um fit mínimo em
    `device="cuda"`. Resultado é cacheado para não pagar o smoke test toda vez.
    """
    try:
        import xgboost as xgb
    except ImportError:
        return False
    if not xgb.build_info().get("USE_CUDA"):
        return False
    try:
        X = np.zeros((8, 2), dtype=np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        xgb.XGBClassifier(
            device="cuda",
            tree_method="hist",
            n_estimators=1,
            verbosity=0,
        ).fit(X, y)
    except Exception:
        return False
    return True


def get_device() -> str:
    """Retorna `"cuda"` se há GPU utilizável, senão `"cpu"`."""
    return "cuda" if cuda_available() else "cpu"
