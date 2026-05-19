"""Seeds, paths e folds compartilhados entre experimentos.

Centraliza tudo que precisa ser idêntico entre runs para garantir comparações
pareadas (mesmos folds, mesmo random_state) e reprodutibilidade.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

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
