"""Métricas, comparações pareadas e plots de avaliação.

Funções de uso direto no `main.ipynb`:

- `paired_compare(scores_a, scores_b)` — Wilcoxon + paired t-test.
- `summary_from_search(search)` — F1-macro = X ± Y com p-valor vs baseline.
- `plot_confusion(y_true, y_pred, classes, title)` — matriz normalizada.
- `plot_roc_pr_macro(y_true, y_score, classes, title)` — ROC + PR macro multiclasse.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .utils import FIGURES, ensure_dirs


@dataclass(frozen=True)
class PairedResult:
    mean_a: float
    mean_b: float
    diff: float
    wilcoxon_p: float
    t_test_p: float
    n: int

    def to_dict(self) -> dict:
        return {
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "diff_b_minus_a": self.diff,
            "wilcoxon_p": self.wilcoxon_p,
            "paired_t_p": self.t_test_p,
            "n_folds": self.n,
        }


def paired_compare(scores_a: list[float], scores_b: list[float]) -> PairedResult:
    """Compara duas listas de scores por fold (baseline vs variante).

    Retorna p-valores de Wilcoxon e paired t-test. Diferenças são (B - A);
    p<0.05 indica que a variante difere significativamente do baseline.
    Robusto a ties (diferenças nulas em todos os pares) — retorna p=1.0.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"scores devem ter o mesmo tamanho ({a.shape} vs {b.shape})")
    diffs = b - a
    if np.all(diffs == 0):
        wilcoxon_p = 1.0
        t_p = 1.0
    else:
        wilcoxon_p = float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)
        t_p = float(stats.ttest_rel(a, b).pvalue)
    return PairedResult(
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        diff=float(diffs.mean()),
        wilcoxon_p=wilcoxon_p,
        t_test_p=t_p,
        n=len(a),
    )


def summary_from_search(search, label: str | None = None) -> dict:
    """F1-macro = mean ± std dos folds da MELHOR configuração."""
    cv_results = search.cv_results_
    best_idx = search.best_index_
    return {
        "label": label or "best",
        "best_params": cv_results["params"][best_idx],
        "f1_macro_mean": float(cv_results["mean_test_score"][best_idx]),
        "f1_macro_std": float(cv_results["std_test_score"][best_idx]),
        "f1_macro_train_mean": float(cv_results["mean_train_score"][best_idx]),
    }


def format_summary(s: dict) -> str:
    return (
        f"{s['label']}: F1-macro = {s['f1_macro_mean']:.4f} ± {s['f1_macro_std']:.4f} "
        f"(treino={s['f1_macro_train_mean']:.4f})"
    )


# ──────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────


def plot_confusion(
    y_true,
    y_pred,
    classes: list[str],
    title: str,
    filename: str | None = None,
) -> None:
    ensure_dirs()
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax, values_format=".2f", cmap="Blues", colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    if filename:
        fig.savefig(FIGURES / filename, dpi=120)
    plt.close(fig)


def plot_roc_pr_macro(
    y_true,
    y_score: np.ndarray,
    classes: list[str],
    title_prefix: str,
    filename_prefix: str | None = None,
) -> dict[str, float]:
    """ROC + PR macro multiclasse. Retorna AUCs macro."""
    ensure_dirs()
    y_bin = label_binarize(y_true, classes=classes)

    fpr, tpr, roc_auc = {}, {}, {}
    prec, rec, pr_auc = {}, {}, {}
    for i, _ in enumerate(classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        prec[i], rec[i], _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        pr_auc[i] = auc(rec[i], prec[i])

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, cls in enumerate(classes):
        ax.plot(fpr[i], tpr[i], label=f"{cls} (AUC={roc_auc[i]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{title_prefix} — ROC macro multiclasse")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if filename_prefix:
        fig.savefig(FIGURES / f"{filename_prefix}__roc.png", dpi=120)
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, cls in enumerate(classes):
        ax.plot(rec[i], prec[i], label=f"{cls} (AUC={pr_auc[i]:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title_prefix} — PR macro multiclasse")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if filename_prefix:
        fig.savefig(FIGURES / f"{filename_prefix}__pr.png", dpi=120)
    plt.close(fig)

    return {
        "roc_auc_macro": float(np.mean(list(roc_auc.values()))),
        "pr_auc_macro": float(np.mean(list(pr_auc.values()))),
    }


def full_report(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    """Tabela de métricas por classe + médias macro."""
    rep = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    return pd.DataFrame(rep).T.assign(
        balanced_accuracy=[balanced_accuracy_score(y_true, y_pred)] + [np.nan] * (
            len(pd.DataFrame(rep).T) - 1
        ),
        f1_macro=[f1_score(y_true, y_pred, average="macro", zero_division=0)] + [np.nan] * (
            len(pd.DataFrame(rep).T) - 1
        ),
    )
