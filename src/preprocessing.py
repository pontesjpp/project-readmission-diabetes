"""Pipelines de pré-processamento.

`build_baseline_pipeline()` retorna um Pipeline sklearn com o pré-processamento
mínimo exigido (Seção 7.4 das exigências): limpeza estrutural, imputação,
codificação OHE e escalonamento. Pode ser usado como passo de pré-processamento
dentro de qualquer modelo (Pipeline final = preproc → model), evitando leakage
entre folds.

As variantes (balanceamento, escalas, codificações, FE) serão definidas
construindo Pipelines diferentes que reaproveitam estes blocos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from .utils import RANDOM_STATE

COLS_DROP = ["weight", "examide", "citoglipton", "encounter_id", "patient_nbr"]

NUMERIC_COLS: list[str] = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

MEDICATION_COLS: list[str] = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

OTHER_CATEGORICAL_COLS: list[str] = [
    "race", "gender", "age", "max_glu_serum", "A1Cresult",
    "diag_1", "diag_2", "diag_3",
    "payer_code", "medical_specialty",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "change", "diabetesMed",
]

ICD_GROUPS: list[tuple[tuple[int, int], str]] = [
    ((390, 459), "Circulatory"),   ((785, 785), "Circulatory"),
    ((460, 519), "Respiratory"),   ((786, 786), "Respiratory"),
    ((520, 579), "Digestive"),     ((787, 787), "Digestive"),
    ((249, 250), "Diabetes"),
    ((800, 999), "Injury"),
    ((710, 739), "Musculoskeletal"),
    ((580, 629), "Genitourinary"), ((788, 788), "Genitourinary"),
    ((140, 239), "Neoplasms"),
]


def _group_icd(code: object) -> str:
    """Agrupa um código ICD-9 em uma das 9 categorias clínicas principais."""
    s = str(code).strip()
    if not s or s in ("nan", "?", "None"):
        return "Other"
    if s[0] in ("V", "E"):
        return "Other"
    try:
        n = float(s.split(".")[0])
    except ValueError:
        return "Other"
    for (lo, hi), label in ICD_GROUPS:
        if lo <= n <= hi:
            return label
    return "Other"


class RawCleaner(BaseEstimator, TransformerMixin):
    """Limpeza estrutural sem parâmetros aprendidos.

    - Remove colunas inúteis (weight, examide, citoglipton, IDs).
    - Substitui `?` por NaN.
    - Agrupa diag_1/2/3 em 9 categorias clínicas.
    - Converte IDs categóricos (admission_type_id etc.) para string.

    Como é determinístico, pode rodar uma vez fora do CV sem leakage —
    mas como Transformer permite encadear num Pipeline.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.drop(columns=COLS_DROP, errors="ignore").copy()
        df = df.replace("?", np.nan)
        for col in ("diag_1", "diag_2", "diag_3"):
            if col in df.columns:
                df[col] = df[col].apply(_group_icd)
        for col in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df


HIGH_CARD_CATEGORICAL_COLS: list[str] = [
    "payer_code",
    "medical_specialty",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

LOW_CARD_CATEGORICAL_COLS: list[str] = [
    c for c in OTHER_CATEGORICAL_COLS if c not in HIGH_CARD_CATEGORICAL_COLS
]


def _make_column_transformer(scaler: str = "standard") -> ColumnTransformer:
    """Numérico → impute+scale; Categórico → impute+OHE.

    `scaler` ∈ {"standard", "robust"} controla o escalonamento numérico.
    Usa `handle_unknown='ignore'` no OHE para tolerar categorias novas no teste.
    """
    scaler_obj = RobustScaler() if scaler == "robust" else StandardScaler()
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", scaler_obj),
    ])
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    categorical_cols = MEDICATION_COLS + OTHER_CATEGORICAL_COLS
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_target_encoding_transformer() -> ColumnTransformer:
    """Categóricas de alta cardinalidade via TargetEncoder; resto via OHE.

    TargetEncoder do sklearn faz CV interno por categoria (anti-leakage). Ainda
    assim, o ColumnTransformer continua sendo ajustado dentro do CV externo do
    GridSearchCV, então não há vazamento entre folds.
    """
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    ohe_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    te_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("te", TargetEncoder(target_type="multiclass", random_state=RANDOM_STATE)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("te_cat", te_pipe, HIGH_CARD_CATEGORICAL_COLS),
            ("ohe_cat", ohe_pipe, MEDICATION_COLS + LOW_CARD_CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_baseline_pipeline() -> Pipeline:
    """Pipeline de pré-processamento BASELINE.

    Use como prefixo de qualquer modelo:

        from sklearn.pipeline import Pipeline
        full = Pipeline([
            ("prep", build_baseline_pipeline()),
            ("clf",  modelo_de_alguma_familia),
        ])
    """
    return Pipeline(steps=[
        ("clean", RawCleaner()),
        ("ct", _make_column_transformer(scaler="standard")),
    ])


def build_robust_pipeline() -> Pipeline:
    """Variante: troca StandardScaler por RobustScaler (resistente a outliers)."""
    return Pipeline(steps=[
        ("clean", RawCleaner()),
        ("ct", _make_column_transformer(scaler="robust")),
    ])


def build_target_encoding_pipeline() -> Pipeline:
    """Variante: TargetEncoder nas categóricas de alta cardinalidade."""
    return Pipeline(steps=[
        ("clean", RawCleaner()),
        ("ct", _make_target_encoding_transformer()),
    ])


def build_smote_pipeline():
    """Variante: pré-processamento baseline + SMOTE.

    Retorna um `imblearn.pipeline.Pipeline` para que o SMOTE seja aplicado
    DENTRO de cada fold de CV (sem leakage). `src.models._wrap` detecta o
    tipo e mantém a estrutura ao anexar o classificador.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    return ImbPipeline(steps=[
        ("clean", RawCleaner()),
        ("ct", _make_column_transformer(scaler="standard")),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
    ])
