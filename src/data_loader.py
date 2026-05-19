"""Carga + split isolado do dataset Diabetes 130-US Hospitals.

Implementa a regra inegociável: split estratificado feito uma única vez,
persistido em `data/interim/`, e o teste só é liberado mediante o token
`I_AM_IN_FINAL_EVALUATION`.

Uso típico:

    from src.data_loader import load_train
    X_train, y_train = load_train()

E na seção 9 do main.ipynb (apenas lá):

    from src.data_loader import load_test, UNLOCK_TOKEN
    X_test, y_test = load_test(UNLOCK_TOKEN)
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import DATA_INTERIM, DATA_RAW, RANDOM_STATE, ensure_dirs

UNLOCK_TOKEN = "I_AM_IN_FINAL_EVALUATION"
UCI_DATASET_ID = 296
TEST_SIZE = 0.2

TRAIN_X_PATH = DATA_INTERIM / "X_train.parquet"
TRAIN_Y_PATH = DATA_INTERIM / "y_train.parquet"
TEST_X_PATH = DATA_INTERIM / "X_test.parquet"
TEST_Y_PATH = DATA_INTERIM / "y_test.parquet"
RAW_PATH = DATA_RAW / "diabetic_data.parquet"


def _fetch_raw() -> tuple[pd.DataFrame, pd.Series]:
    """Baixa o dataset 296 do UCI ML Repository e cacheia em data/raw/.

    Usa `ucimlrepo` na primeira chamada; depois lê o parquet local.
    """
    ensure_dirs()
    if RAW_PATH.exists():
        df = pd.read_parquet(RAW_PATH)
        return df.drop(columns=["readmitted"]), df["readmitted"]

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()
    y.name = "readmitted"

    combined = X.copy()
    combined["readmitted"] = y.values
    combined.to_parquet(RAW_PATH, index=False)
    return X, y


def _materialize_split() -> None:
    """Cria o split estratificado treino/teste e persiste em data/interim/."""
    ensure_dirs()
    X, y = _fetch_raw()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_train.to_parquet(TRAIN_X_PATH, index=False)
    X_test.to_parquet(TEST_X_PATH, index=False)
    y_train.to_frame().to_parquet(TRAIN_Y_PATH, index=False)
    y_test.to_frame().to_parquet(TEST_Y_PATH, index=False)


def _ensure_split() -> None:
    if not (TRAIN_X_PATH.exists() and TEST_X_PATH.exists()):
        _materialize_split()


def load_train() -> tuple[pd.DataFrame, pd.Series]:
    """Carrega APENAS o split de treino. Pode ser chamado em qualquer seção."""
    _ensure_split()
    X = pd.read_parquet(TRAIN_X_PATH)
    y = pd.read_parquet(TRAIN_Y_PATH).iloc[:, 0]
    return X, y


def load_test(unlock_token: str) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega o split de teste — SÓ na avaliação final (seção 9 do main.ipynb).

    Exige o token literal `I_AM_IN_FINAL_EVALUATION`. Qualquer outro valor
    levanta `PermissionError` para impedir leakage acidental.
    """
    if unlock_token != UNLOCK_TOKEN:
        raise PermissionError(
            "Acesso ao conjunto de teste bloqueado. "
            "Use load_test('I_AM_IN_FINAL_EVALUATION') apenas na seção 9 do main.ipynb."
        )
    _ensure_split()
    X = pd.read_parquet(TEST_X_PATH)
    y = pd.read_parquet(TEST_Y_PATH).iloc[:, 0]
    return X, y


def split_info() -> dict:
    """Metadados do split — tamanhos e distribuição de classes no TREINO.

    Não retorna nem inspeciona os rótulos de teste para não vazar informação.
    """
    _ensure_split()
    X_train = pd.read_parquet(TRAIN_X_PATH)
    y_train = pd.read_parquet(TRAIN_Y_PATH).iloc[:, 0]
    n_test = len(pd.read_parquet(TEST_X_PATH, columns=[X_train.columns[0]]))
    return {
        "train_shape": X_train.shape,
        "test_n_rows": n_test,
        "n_features": X_train.shape[1],
        "class_balance_train": y_train.value_counts(normalize=True).to_dict(),
    }
