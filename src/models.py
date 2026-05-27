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

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from .utils import RANDOM_STATE, get_device


# Compat sklvq 0.1.2 × sklearn 1.8: aplicado no IMPORT do módulo. Joblib workers
# que precisem unpicklar `PatchedGLVQ` reimportam src.models, então o side-effect
# do patch vai junto — diferente do monkey-patch tardio que era process-local.
def _apply_sklvq_sklearn_compat() -> None:
    try:
        import sklvq.models._base as _lvq_base
    except Exception:
        return
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

    if not getattr(_lvq_base.LVQBaseClass, "_sklvq_compat_applied", False):
        _lvq_base.LVQBaseClass._validate_data = _validate_data_compat
        _lvq_base.check_array = _check_array_compat
        _lvq_base.LVQBaseClass._sklvq_compat_applied = True


_apply_sklvq_sklearn_compat()

PrepFactory = Callable[[], Any]
ModelFactory = Callable[[PrepFactory], Pipeline]


class MLPStringLabel(MLPClassifier):
    """MLP que aceita rótulos string sem disparar o bug de early_stopping.

    `MLPClassifier(early_stopping=True)` chama `np.isnan(y_pred)` sobre o
    conjunto de validação interno; com rótulos string (`<30`/`>30`/`NO`) o
    ufunc estoura `TypeError`. Encodamos no fit e desfazemos no predict.
    Precisa ser top-level (não local) para sobreviver ao pickle do joblib
    quando o estimator é clonado para workers (`n_jobs>1`).
    """

    def fit(self, X, y, **kw):
        self._le = LabelEncoder()
        return super().fit(X, self._le.fit_transform(y), **kw)

    def predict(self, X):
        return self._le.inverse_transform(super().predict(X))


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
    """Pipeline = preprocessador (recriado por chamada) → estimator.

    Se a fábrica devolver um `imblearn.pipeline.Pipeline` (variante SMOTE/etc.),
    achatamos os steps no nível do pipeline e mantemos o tipo do imblearn —
    necessário para que o resampler seja aplicado por fold sem leakage.
    """
    prep = prep_factory()
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline

        if isinstance(prep, ImbPipeline):
            return ImbPipeline(steps=list(prep.steps) + [("clf", estimator)])
    except ImportError:
        pass
    return Pipeline(steps=[("prep", prep), ("clf", estimator)])


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
    return _wrap(
        prep_factory,
        MLPStringLabel(
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
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    class _XGBStringLabel(XGBClassifier):
        """XGBClassifier wrapper that accepts string class labels."""

        def fit(self, X, y, **kw):
            self._le = LabelEncoder()
            return super().fit(X, self._le.fit_transform(y), **kw)

        def predict(self, X):
            return self._le.inverse_transform(super().predict(X))

        def predict_proba(self, X):
            return super().predict_proba(X)

    return _wrap(
        prep_factory,
        _XGBStringLabel(
            random_state=RANDOM_STATE,
            n_jobs=2,
            tree_method="hist",
            device=get_device(),
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
# LVQ (sklvq) — via PatchedGLVQ
# ──────────────────────────────────────────────────────────────────────────
# Compat patch é aplicado no topo do módulo (sobrevive ao import dos workers
# do joblib). PatchedGLVQ vive aqui em src.models — ao unpicklar, joblib
# precisa importar src.models, disparando o patch automaticamente.

try:
    from sklvq import GLVQ as _GLVQ

    class PatchedGLVQ(_GLVQ):
        """GLVQ com compat shim de sklearn 1.8 aplicado no import de src.models."""

        pass

    _LVQ_AVAILABLE = True
except Exception:
    _LVQ_AVAILABLE = False


def _make_lvq(prep_factory: PrepFactory) -> Pipeline:
    if not _LVQ_AVAILABLE:
        raise RuntimeError("sklvq não disponível ou patch falhou.")
    return _wrap(
        prep_factory,
        PatchedGLVQ(
            random_state=RANDOM_STATE,
            prototype_init="class-conditional-mean",
            prototype_n_per_class=1,
        ),
    )


# Grid enxuto: LVQ é muito caro em ~80k × ~100+ features OHE.
LVQ_GRID: dict[str, list[Any]] = {
    "clf__prototype_n_per_class": [1, 2],
    "clf__activation_type": ["sigmoid"],
}


# ──────────────────────────────────────────────────────────────────────────
# Comitê de RNAs (bagging de MLPs)
# ──────────────────────────────────────────────────────────────────────────


def _make_mlp_committee(prep_factory: PrepFactory) -> Pipeline:
    from sklearn.ensemble import BaggingClassifier

    base_mlp = MLPStringLabel(
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
    "clf__n_estimators": [5, 10],
    "clf__max_samples": [1.0],
    "clf__estimator__hidden_layer_sizes": [(64,)],
    "clf__estimator__alpha": [1e-4, 1e-2],
}


# ──────────────────────────────────────────────────────────────────────────
# Stacking heterogêneo
# ──────────────────────────────────────────────────────────────────────────


def _make_stacking(prep_factory: PrepFactory) -> Pipeline:
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    # Estimadores base intencionalmente leves: o stacking original com
    # RF/LGBM=200 árvores + cv interno=3 levava ~4h/grid. Reduzir é
    # documentado: ainda assim heterogêneo (árvore profunda + boosting +
    # bagging) e suficiente para a comparação metodológica da rubrica.
    estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=80,
                max_depth=12,
                n_jobs=2,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "lgbm",
            LGBMClassifier(
                n_estimators=100,
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


# Grid enxuto: 3 candidatos (variação só no meta-classificador).
STACKING_GRID: dict[str, list[Any]] = {
    "clf__final_estimator__C": [0.1, 1.0, 10.0],
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
        enabled_for_checkpoint=True,
        notes="XGBoost com tree_method=hist.",
    ),
    "lvq": ModelSpec(
        name="lvq",
        make_pipeline=_make_lvq,
        param_grid=LVQ_GRID,
        search="grid",
        notes="GLVQ (sklvq) — protótipos por classe; grid enxuto pelo custo em alta dim.",
    ),
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
