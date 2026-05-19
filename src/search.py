"""Busca de hiperparâmetros + curvas treino vs validação.

`run_search(spec, X, y, prep_factory)` executa Grid ou Randomized SearchCV
seguindo o que está declarado em `ModelSpec`, retorna o `search` ajustado e
persiste:

- `reports/cv_results/<modelo>__<tag>.csv` — `cv_results_` completo
- `reports/figures/<modelo>__<tag>__train_vs_val.png` — curva por configuração

Os scores por fold (split0_test_score, split1_test_score, …) ficam no CSV
e são consumidos por `evaluation.paired_compare` para Wilcoxon/paired t-test.
"""

from __future__ import annotations

from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .models import ModelSpec
from .utils import CV_RESULTS, FIGURES, RANDOM_STATE, ensure_dirs, get_cv

PrepFactory = Callable[[], Any]


def _build_search(spec: ModelSpec, estimator, cv) -> GridSearchCV | RandomizedSearchCV:
    common = dict(
        scoring="f1_macro",
        cv=cv,
        n_jobs=2,
        return_train_score=True,
        refit=True,
        verbose=1,
    )
    if spec.search == "grid":
        return GridSearchCV(estimator=estimator, param_grid=spec.param_grid, **common)
    if spec.search == "randomized":
        return RandomizedSearchCV(
            estimator=estimator,
            param_distributions=spec.param_grid,
            n_iter=spec.n_iter,
            random_state=RANDOM_STATE,
            **common,
        )
    raise ValueError(f"Estratégia desconhecida: {spec.search!r}")


def _plot_train_vs_val(cv_results: dict, spec: ModelSpec, tag: str) -> None:
    """Curva treino vs validação por configuração (ordenada por mean_test_score)."""
    df = pd.DataFrame(cv_results)
    df = df.sort_values("mean_test_score").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = range(len(df))
    ax.errorbar(
        x, df["mean_train_score"], yerr=df["std_train_score"],
        marker="o", label="Treino (CV)", capsize=2,
    )
    ax.errorbar(
        x, df["mean_test_score"], yerr=df["std_test_score"],
        marker="s", label="Validação (CV)", capsize=2,
    )
    ax.set_xlabel("Configuração de HP (ordenada por F1-macro de validação)")
    ax.set_ylabel("F1-macro")
    ax.set_title(f"{spec.name} — curva treino vs validação ({tag})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGURES / f"{spec.name}__{tag}__train_vs_val.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run_search(
    spec: ModelSpec,
    X,
    y,
    prep_factory: PrepFactory,
    tag: str = "baseline",
):
    """Executa a busca de HP descrita em `spec` e persiste artefatos.

    Returns
    -------
    search : ajustado, com `best_estimator_` pronto para a etapa de comparação.
    """
    ensure_dirs()
    estimator = spec.make_pipeline(prep_factory)
    cv = get_cv()
    search = _build_search(spec, estimator, cv)
    search.fit(X, y)

    cv_results = search.cv_results_
    pd.DataFrame(cv_results).to_csv(
        CV_RESULTS / f"{spec.name}__{tag}.csv", index=False
    )
    _plot_train_vs_val(cv_results, spec, tag)
    return search


def best_per_fold_scores(search) -> list[float]:
    """Retorna os scores por fold (F1-macro) da MELHOR configuração.

    Esses valores são a entrada do Wilcoxon pareado entre baseline e variante.
    """
    cv_results = search.cv_results_
    best_idx = search.best_index_
    n_folds = sum(
        1 for k in cv_results if k.startswith("split") and k.endswith("_test_score")
    )
    return [cv_results[f"split{i}_test_score"][best_idx] for i in range(n_folds)]
