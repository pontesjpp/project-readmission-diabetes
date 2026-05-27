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

import ast
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .utils import CV_RESULTS, FIGURES, RANDOM_STATE, ensure_dirs, get_cv

if TYPE_CHECKING:
    from .models import ModelSpec
else:
    from .models import ModelSpec  # noqa: F401  (runtime import for ModelSpec hints)

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


@dataclass
class CachedSearch:
    """Reconstrução de um search a partir do CSV persistido em `cv_results`.

    Tem o mínimo para a etapa de comparação (cv_results_, best_index_, best_score_,
    best_params_). NÃO traz `best_estimator_` — para reusar o modelo treinado,
    chamar `refit_best(...)`.
    """

    cv_results_: dict[str, Any]
    best_index_: int
    best_score_: float
    best_params_: dict[str, Any]


_NUMPY_WRAP_RE = re.compile(r"np\.(?:float\d*|int\d*|bool_)\(([^()]+)\)")


def _parse_params_cell(s: str) -> dict:
    """Robusto a `np.float64(10.0)` que `ast.literal_eval` não engole.

    Substitui chamadas `np.<type>(<lit>)` por `<lit>` antes do literal_eval.
    Mantém comportamento padrão pra dicts puros.
    """
    if not isinstance(s, str):
        return s
    cleaned = _NUMPY_WRAP_RE.sub(r"\1", s)
    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        # último recurso: eval com namespace mínimo (CSVs gerados por nós, controlados)
        import numpy as _np

        return eval(s, {"__builtins__": {}}, {"np": _np})


def load_search_from_csv(spec: "ModelSpec", tag: str) -> CachedSearch | None:
    """Lê `cv_results__<modelo>__<tag>.csv` e devolve um CachedSearch ou None."""
    csv = CV_RESULTS / f"{spec.name}__{tag}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if df.empty or "mean_test_score" not in df.columns:
        return None
    cv_results = {col: df[col].to_numpy() for col in df.columns}
    if "params" in df.columns:
        cv_results["params"] = [_parse_params_cell(p) for p in df["params"].tolist()]
    best_idx = int(df["mean_test_score"].idxmax())
    return CachedSearch(
        cv_results_=cv_results,
        best_index_=best_idx,
        best_score_=float(df["mean_test_score"].iloc[best_idx]),
        best_params_=cv_results["params"][best_idx] if "params" in cv_results else {},
    )


def get_or_run_search(
    spec: "ModelSpec",
    X,
    y,
    prep_factory: PrepFactory,
    tag: str = "baseline",
    force: bool = False,
):
    """Carrega o search do CSV se já existir; senão executa `run_search`.

    Permite reprodutibilidade do `main.ipynb` sem refazer horas de busca:
    a primeira execução popula `reports/cv_results/`, e execuções subsequentes
    reutilizam o cache. Use `force=True` para sobrescrever.
    """
    if not force:
        cached = load_search_from_csv(spec, tag)
        if cached is not None:
            return cached
    return run_search(spec, X, y, prep_factory, tag=tag)


def refit_best(spec: "ModelSpec", X, y, prep_factory: PrepFactory, best_params: dict):
    """Refit do modelo vencedor com os hiperparâmetros escolhidos.

    Usado na seção 9 (avaliação final) sem precisar manter o `best_estimator_`
    do GridSearchCV em memória. Devolve um Pipeline ajustado em (X, y).
    """
    pipe = spec.make_pipeline(prep_factory)
    pipe.set_params(**best_params)
    pipe.fit(X, y)
    return pipe
