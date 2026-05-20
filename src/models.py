"""Factories dos 10 algoritmos obrigatórios.

Cada entrada em `MODEL_REGISTRY` define:
- `name`: chave curta para logs e arquivos
- `make_pipeline`: função que recebe o pré-processador e retorna um Pipeline
- `param_grid`: dicionário compatível com Grid/RandomizedSearchCV
- `search`: "grid" ou "randomized"
- `n_iter`: número de iterações para Randomized (ignorado em Grid)
- `enabled_for_checkpoint`: se True, é executado na rodada do checkpoint (20/05)

Para o checkpoint usamos apenas 3 modelos rápidos. Os demais ficam registrados
para que a fase final só precise alterar a flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .utils import RANDOM_STATE

PrepFactory = Callable[[], Any]
ModelFactory = Callable[[PrepFactory], Pipeline]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    make_pipeline: ModelFactory
    param_grid: dict[str, list[Any]]
    search: str = "grid"
    n_iter: int = 10
    enabled_for_checkpoint: bool = False
    notes: str = ""


def _wrap(prep_factory: PrepFactory, estimator) -> Pipeline:
    """Pipeline = preprocessador (recriado por chamada) → estimator."""
    return Pipeline(steps=[("prep", prep_factory()), ("clf", estimator)])


# ──────────────────────────────────────────────────────────────────────────
# Modelos habilitados no checkpoint
# ──────────────────────────────────────────────────────────────────────────


def _make_decision_tree(prep_factory: PrepFactory) -> Pipeline:
    return _wrap(prep_factory, DecisionTreeClassifier(random_state=RANDOM_STATE))


DT_GRID: dict[str, list[Any]] = {
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": [6, 12, 20],
    "clf__min_samples_leaf": [1, 10],
}


def _make_random_forest(prep_factory: PrepFactory) -> Pipeline:
    return _wrap(
        prep_factory,
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
    )


RF_GRID: dict[str, list[Any]] = {
    "clf__n_estimators": [200],
    "clf__max_depth": [12, 20, None],
    "clf__max_features": ["sqrt", "log2"],
    "clf__min_samples_leaf": [1, 5],
    "clf__class_weight": [None, "balanced"],
}


def _make_lightgbm(prep_factory: PrepFactory) -> Pipeline:
    from lightgbm import LGBMClassifier

    return _wrap(
        prep_factory,
        LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=2,
            objective="multiclass",
            verbose=-1,
        ),
    )


LGBM_GRID: dict[str, list[Any]] = {
    "clf__num_leaves": [31, 63],
    "clf__learning_rate": [0.05, 0.1],
    "clf__n_estimators": [200, 400],
    "clf__min_data_in_leaf": [20, 50],
}


# ──────────────────────────────────────────────────────────────────────────
# Demais modelos — registrados mas desativados no checkpoint
# ──────────────────────────────────────────────────────────────────────────


def _make_knn(prep_factory: PrepFactory) -> Pipeline:
    return _wrap(prep_factory, KNeighborsClassifier(n_jobs=2))


KNN_GRID: dict[str, list[Any]] = {
    "clf__n_neighbors": [5, 11, 21, 31],
    "clf__weights": ["uniform", "distance"],
    "clf__metric": ["euclidean", "manhattan"],
}


def _make_svm(prep_factory: PrepFactory) -> Pipeline:
    from sklearn.svm import LinearSVC

    return _wrap(
        prep_factory,
        LinearSVC(random_state=RANDOM_STATE, dual="auto", max_iter=3000),
    )


SVM_GRID: dict[str, list[Any]] = {
    "clf__C": list(np.logspace(-2, 2, 5)),
    "clf__class_weight": [None, "balanced"],
}


def _make_mlp(prep_factory: PrepFactory) -> Pipeline:
    from sklearn.neural_network import MLPClassifier

    return _wrap(
        prep_factory,
        MLPClassifier(
            random_state=RANDOM_STATE,
            early_stopping=True,
            max_iter=200,
        ),
    )


MLP_GRID: dict[str, list[Any]] = {
    "clf__hidden_layer_sizes": [(64,), (128,), (64, 64), (128, 64)],
    "clf__activation": ["relu", "tanh"],
    "clf__alpha": list(np.logspace(-5, -1, 5)),
    "clf__learning_rate_init": list(np.logspace(-4, -1, 4)),
}


def _make_xgboost(prep_factory: PrepFactory) -> Pipeline:
    from xgboost import XGBClassifier

    return _wrap(
        prep_factory,
        XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=2,
            tree_method="hist",
            eval_metric="mlogloss",
        ),
    )


XGB_GRID: dict[str, list[Any]] = {
    "clf__n_estimators": [200, 500, 800],
    "clf__max_depth": [4, 6, 8, 10],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__subsample": [0.7, 0.9, 1.0],
    "clf__colsample_bytree": [0.7, 0.9, 1.0],
}


# ──────────────────────────────────────────────────────────────────────────
# LVQ (sklvq) — EM STANDBY
# ──────────────────────────────────────────────────────────────────────────
# sklvq 0.1.2 usa internals do sklearn removidos em 1.6+ (`_validate_data`) e
# o argumento `force_all_finite` renomeado para `ensure_all_finite`. O shim
# abaixo funciona em single-process mas é process-local: joblib (n_jobs>1)
# spawna workers que reimportam sklvq sem o patch, fazendo a busca falhar.
# Solução pendente: trocar por um subclass `PatchedGLVQ(GLVQ)` definido aqui
# em `src.models` para que a correção viaje junto no unpickle. Não removido
# do registry como "stub" para evitar perder a contagem de 10 algoritmos
# antes da entrega final (10/06) — a entrada está comentada em MODEL_REGISTRY.

_LVQ_PATCH_APPLIED = False


def _apply_sklvq_sklearn_compat() -> None:
    global _LVQ_PATCH_APPLIED
    if _LVQ_PATCH_APPLIED:
        return
    import sklvq.models._base as _lvq_base
    from sklearn.utils import check_array as _check_array
    from sklearn.utils.validation import validate_data as _validate_data

    def _validate_data_compat(self, *args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _validate_data(self, *args, **kwargs)

    def _check_array_compat(*args, **kwargs):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return _check_array(*args, **kwargs)

    _lvq_base.LVQBaseClass._validate_data = _validate_data_compat
    _lvq_base.check_array = _check_array_compat
    _LVQ_PATCH_APPLIED = True


def _make_lvq(prep_factory: PrepFactory) -> Pipeline:
    _apply_sklvq_sklearn_compat()
    from sklvq import GLVQ

    return _wrap(
        prep_factory,
        GLVQ(
            random_state=RANDOM_STATE,
            prototype_init="class-conditional-mean",
            prototype_n_per_class=1,
        ),
    )


LVQ_GRID: dict[str, list[Any]] = {
    "clf__prototype_n_per_class": [1, 2, 3],
    "clf__activation_type": ["sigmoid", "identity"],
    "clf__distance_type": ["squared-euclidean"],
}


# ──────────────────────────────────────────────────────────────────────────
# Comitê de RNAs (bagging de MLPs)
# ──────────────────────────────────────────────────────────────────────────


def _make_mlp_committee(prep_factory: PrepFactory) -> Pipeline:
    from sklearn.ensemble import BaggingClassifier
    from sklearn.neural_network import MLPClassifier

    base_mlp = MLPClassifier(
        random_state=RANDOM_STATE,
        early_stopping=True,
        max_iter=200,
        hidden_layer_sizes=(64,),
    )
    return _wrap(
        prep_factory,
        BaggingClassifier(
            estimator=base_mlp,
            random_state=RANDOM_STATE,
            n_estimators=10,
            n_jobs=2,
        ),
    )


MLP_COMMITTEE_GRID: dict[str, list[Any]] = {
    "clf__n_estimators": [5, 10, 15],
    "clf__max_samples": [0.6, 0.8, 1.0],
    "clf__max_features": [0.7, 1.0],
    "clf__estimator__hidden_layer_sizes": [(64,), (128,), (64, 64)],
    "clf__estimator__alpha": list(np.logspace(-5, -2, 4)),
}


# ──────────────────────────────────────────────────────────────────────────
# Stacking heterogêneo
# ──────────────────────────────────────────────────────────────────────────


def _make_stacking(prep_factory: PrepFactory) -> Pipeline:
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                n_jobs=2,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "lgbm",
            LGBMClassifier(
                n_estimators=200,
                n_jobs=2,
                random_state=RANDOM_STATE,
                objective="multiclass",
                verbose=-1,
            ),
        ),
        (
            "dt",
            DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        ),
    ]
    final = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    return _wrap(
        prep_factory,
        StackingClassifier(
            estimators=estimators,
            final_estimator=final,
            cv=3,
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=1,
        ),
    )


STACKING_GRID: dict[str, list[Any]] = {
    "clf__final_estimator__C": [0.1, 1.0, 10.0],
    "clf__passthrough": [False, True],
    "clf__stack_method": ["predict_proba", "auto"],
}


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "decision_tree": ModelSpec(
        name="decision_tree",
        make_pipeline=_make_decision_tree,
        param_grid=DT_GRID,
        search="grid",
        enabled_for_checkpoint=True,
        notes="Árvore de Decisão (CART, gini/entropy).",
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        make_pipeline=_make_random_forest,
        param_grid=RF_GRID,
        search="randomized",
        n_iter=6,
        enabled_for_checkpoint=True,
        notes="Random Forest (com e sem class_weight balanceado).",
    ),
    "lightgbm": ModelSpec(
        name="lightgbm",
        make_pipeline=_make_lightgbm,
        param_grid=LGBM_GRID,
        search="randomized",
        n_iter=6,
        enabled_for_checkpoint=True,
        notes="LightGBM (gradient boosting de árvores, rápido).",
    ),
    "knn": ModelSpec(
        name="knn",
        make_pipeline=_make_knn,
        param_grid=KNN_GRID,
        search="grid",
        notes="K-NN — pesado nesse dataset; ativar após checkpoint.",
    ),
    "svm": ModelSpec(
        name="svm",
        make_pipeline=_make_svm,
        param_grid=SVM_GRID,
        search="randomized",
        n_iter=5,
        notes="LinearSVC para escalar ao tamanho do dataset.",
    ),
    "mlp": ModelSpec(
        name="mlp",
        make_pipeline=_make_mlp,
        param_grid=MLP_GRID,
        search="randomized",
        n_iter=10,
        notes="MLP com early stopping.",
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        make_pipeline=_make_xgboost,
        param_grid=XGB_GRID,
        search="randomized",
        n_iter=10,
        notes="XGBoost com tree_method=hist.",
    ),
    # "lvq": EM STANDBY — sklvq 0.1.2 incompatível com sklearn 1.8 em n_jobs>1.
    # Pendente: implementar `PatchedGLVQ(GLVQ)` em src.models e reativar aqui.
    # "lvq": ModelSpec(
    #     name="lvq",
    #     make_pipeline=_make_lvq,
    #     param_grid=LVQ_GRID,
    #     search="grid",
    #     notes="GLVQ (sklvq) — protótipos por classe; sensível à dimensionalidade do OHE.",
    # ),
    "mlp_committee": ModelSpec(
        name="mlp_committee",
        make_pipeline=_make_mlp_committee,
        param_grid=MLP_COMMITTEE_GRID,
        search="randomized",
        n_iter=8,
        notes="Comitê de MLPs via BaggingClassifier (bootstrap de amostras e features).",
    ),
    "stacking": ModelSpec(
        name="stacking",
        make_pipeline=_make_stacking,
        param_grid=STACKING_GRID,
        search="grid",
        notes="Stacking heterogêneo: RF + LightGBM + DT, meta-learner LogisticRegression.",
    ),
}


def checkpoint_models() -> list[ModelSpec]:
    """Retorna apenas os modelos marcados para o checkpoint."""
    return [m for m in MODEL_REGISTRY.values() if m.enabled_for_checkpoint]
