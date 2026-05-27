"""Avaliação final no teste — Seção 9 do workflow.

Refit do melhor modelo (selecionado a partir dos cv_results persistidos)
em TODO o X_train, predição em X_test, e geração de:

- reports/cv_results/final_test_results.csv — métricas por classe + macro
- reports/figures/final_confusion.png — matriz de confusão normalizada
- reports/figures/final__roc.png — ROC macro multiclasse
- reports/figures/final__pr.png — PR macro multiclasse

Uso:
    uv run python scripts/run_final_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.data_loader import UNLOCK_TOKEN, load_test, load_train
from src.evaluation import full_report, plot_confusion, plot_roc_pr_macro
from src.models import MODEL_REGISTRY
from src.preprocessing import (
    build_baseline_pipeline,
    build_robust_pipeline,
    build_smote_pipeline,
    build_target_encoding_pipeline,
    build_undersampling_pipeline,
)
from src.search import best_per_fold_scores, load_search_from_csv, refit_best
from src.utils import CV_RESULTS, FIGURES, ensure_dirs, set_global_seed

CLASSES = ["<30", ">30", "NO"]

VARIANT_FACTORIES = {
    "baseline": build_baseline_pipeline,
    "smote": build_smote_pipeline,
    "undersample": build_undersampling_pipeline,
    "robust": build_robust_pipeline,
    "target_enc": build_target_encoding_pipeline,
}

# Tags de variantes a considerar no ranking
VARIANT_TAGS = list(VARIANT_FACTORIES.keys())


def _build_ranking() -> pd.DataFrame:
    """Reconstrói o ranking de todas as configurações (modelo × variante)
    a partir dos CSVs persistidos em reports/cv_results/."""
    rows: list[dict] = []
    for spec in MODEL_REGISTRY.values():
        for tag in VARIANT_TAGS:
            cached = load_search_from_csv(spec, tag)
            if cached is None:
                continue
            best_idx = cached.best_index_
            rows.append({
                "modelo": spec.name,
                "variante": tag,
                "F1_macro_val": float(cached.cv_results_["mean_test_score"][best_idx]),
                "F1_macro_val_std": float(cached.cv_results_["std_test_score"][best_idx]),
                "F1_macro_train": float(cached.cv_results_["mean_train_score"][best_idx]),
                "best_params": cached.best_params_,
            })
    df = (
        pd.DataFrame(rows)
        .assign(gap_train_val=lambda d: d["F1_macro_train"] - d["F1_macro_val"])
        .sort_values(["F1_macro_val", "F1_macro_val_std"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return df


def main() -> int:
    set_global_seed()
    ensure_dirs()

    # ── 1. Determinar o vencedor a partir do ranking ─────────────────────
    print("=" * 70)
    print("AVALIAÇÃO FINAL NO TESTE (Seção 9)")
    print("=" * 70)

    df_rank = _build_ranking()
    df_rank.to_csv(CV_RESULTS / "ranking_all_configs.csv", index=False)

    display_cols = ["modelo", "variante", "F1_macro_val", "F1_macro_val_std", "gap_train_val"]
    print("\nTop 10 configurações por F1-macro de validação:")
    print(df_rank[display_cols].head(10).to_string(index=False))

    WINNER = df_rank.iloc[0]
    WINNER_MODEL = WINNER["modelo"]
    WINNER_VARIANT = WINNER["variante"]
    WINNER_PARAMS = WINNER["best_params"]

    print(
        f"\n>>> Modelo vencedor: {WINNER_MODEL} | pré-processamento: {WINNER_VARIANT} "
        f"| F1-macro CV = {WINNER['F1_macro_val']:.4f} ± {WINNER['F1_macro_val_std']:.4f}"
    )
    print(f">>> Hiperparâmetros: {WINNER_PARAMS}")

    # ── 2. Refit em todo o treino ────────────────────────────────────────
    spec_winner = MODEL_REGISTRY[WINNER_MODEL]
    factory_winner = VARIANT_FACTORIES[WINNER_VARIANT]

    print(f"\nRefit final: {WINNER_MODEL} | preproc={WINNER_VARIANT}")
    print(f"best_params: {WINNER_PARAMS}")
    print("Treinando em todo X_train...")

    X_train, y_train = load_train()
    best_estimator = refit_best(spec_winner, X_train, y_train, factory_winner, WINNER_PARAMS)

    # ── 3. Predição no teste (único acesso, com token) ───────────────────
    print("\n>>> Carregando conjunto de teste (token UNLOCK)...")
    X_test, y_test = load_test(unlock_token=UNLOCK_TOKEN)

    y_pred = best_estimator.predict(X_test)
    try:
        y_score = best_estimator.predict_proba(X_test)
    except (AttributeError, NotImplementedError):
        y_score = None

    # ── 4. Métricas no teste ─────────────────────────────────────────────
    f1_test = f1_score(y_test, y_pred, average="macro", labels=CLASSES, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"\nF1-macro (TESTE)      = {f1_test:.4f}")
    print(f"F1-macro (CV val)     = {WINNER['F1_macro_val']:.4f} ± {WINNER['F1_macro_val_std']:.4f}")
    print(f"Δ (teste − CV)        = {f1_test - WINNER['F1_macro_val']:+.4f}")
    print(f"Balanced accuracy     = {bal_acc:.4f}")

    # ── 5. Matriz de confusão normalizada ────────────────────────────────
    plot_confusion(
        y_test, y_pred, classes=CLASSES,
        title=f"Avaliação final — {WINNER_MODEL} ({WINNER_VARIANT}) — matriz normalizada",
        filename="final_confusion.png",
    )
    print(f"\n✓ Matriz de confusão salva em {FIGURES / 'final_confusion.png'}")

    # ── 6. ROC + PR macro multiclasse ────────────────────────────────────
    aucs = {}
    if y_score is not None:
        aucs = plot_roc_pr_macro(
            y_test, y_score, classes=CLASSES,
            title_prefix=f"Final ({WINNER_MODEL}|{WINNER_VARIANT})",
            filename_prefix="final",
        )
        print(f"ROC-AUC macro = {aucs['roc_auc_macro']:.4f}")
        print(f"PR-AUC macro  = {aucs['pr_auc_macro']:.4f}")
        print(f"✓ Curvas ROC e PR salvas em {FIGURES}")
    else:
        print("Modelo sem predict_proba; ROC/PR macro pulados.")

    # ── 7. Relatório por classe ──────────────────────────────────────────
    report_df = full_report(y_test, y_pred, classes=CLASSES)
    print("\nMétricas no teste (por classe + médias):")
    print(report_df.round(4).to_string())

    support = report_df.loc[CLASSES, "support"].astype(int)
    recalls = report_df.loc[CLASSES, "recall"]
    print("\nLeitura por classe (custo assimétrico do briefing):")
    for cls in CLASSES:
        print(
            f"  {cls!r:>5} | recall={recalls[cls]:.3f} | support={support[cls]:>5} | "
            f"f1={report_df.loc[cls, 'f1-score']:.3f}"
        )
    print(
        f"\nObs.: a classe '<30' é a de maior custo de erro (intervenção pós-alta perdida). "
        f"recall(<30) = {recalls['<30']:.3f} no teste deve guiar o ponto de operação no deployment."
    )

    # ── 8. Salvar resultados em CSV ──────────────────────────────────────
    results_rows = []
    for cls in CLASSES:
        results_rows.append({
            "class": cls,
            "precision": report_df.loc[cls, "precision"],
            "recall": report_df.loc[cls, "recall"],
            "f1-score": report_df.loc[cls, "f1-score"],
            "support": int(report_df.loc[cls, "support"]),
        })
    results_rows.append({
        "class": "macro_avg",
        "precision": report_df.loc["macro avg", "precision"],
        "recall": report_df.loc["macro avg", "recall"],
        "f1-score": report_df.loc["macro avg", "f1-score"],
        "support": int(report_df.loc["macro avg", "support"]),
    })

    summary_row = {
        "model": WINNER_MODEL,
        "variant": WINNER_VARIANT,
        "best_params": str(WINNER_PARAMS),
        "f1_macro_cv": WINNER["F1_macro_val"],
        "f1_macro_cv_std": WINNER["F1_macro_val_std"],
        "f1_macro_test": f1_test,
        "delta_test_minus_cv": f1_test - WINNER["F1_macro_val"],
        "balanced_accuracy_test": bal_acc,
    }
    if aucs:
        summary_row["roc_auc_macro_test"] = aucs["roc_auc_macro"]
        summary_row["pr_auc_macro_test"] = aucs["pr_auc_macro"]

    # CSV 1: métricas por classe
    pd.DataFrame(results_rows).to_csv(
        CV_RESULTS / "final_test_per_class.csv", index=False
    )
    # CSV 2: resumo do modelo vencedor
    pd.DataFrame([summary_row]).to_csv(
        CV_RESULTS / "final_test_results.csv", index=False
    )

    out_paths = [
        CV_RESULTS / "final_test_per_class.csv",
        CV_RESULTS / "final_test_results.csv",
        CV_RESULTS / "ranking_all_configs.csv",
        FIGURES / "final_confusion.png",
    ]
    if aucs:
        out_paths.extend([FIGURES / "final__roc.png", FIGURES / "final__pr.png"])

    print("\n" + "=" * 70)
    print("ARTEFATOS GERADOS:")
    for p in out_paths:
        print(f"  ✓ {p}")
    print("=" * 70)

    # ── Resumo final ─────────────────────────────────────────────────────
    print(f"\nModelo final em produção: {WINNER_MODEL} | pré-proc: {WINNER_VARIANT}")
    print(f"  F1-macro CV     = {WINNER['F1_macro_val']:.4f} ± {WINNER['F1_macro_val_std']:.4f}")
    print(f"  F1-macro teste  = {f1_test:.4f}  (Δ vs CV: {f1_test - WINNER['F1_macro_val']:+.4f})")
    print(f"  Balanced accuracy teste = {bal_acc:.4f}")
    if aucs:
        print(f"  ROC-AUC macro teste = {aucs['roc_auc_macro']:.4f}")
        print(f"  PR-AUC macro teste  = {aucs['pr_auc_macro']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
