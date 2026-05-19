# Projeto Final — IF1014 Mineração de Dados (CRISP-DM)

Projeto acadêmico de classificação multiclasse usando o dataset **Diabetes 130-US Hospitals (UCI)** para prever readmissão hospitalar. Disciplina IF1014 (UFPE, 2026.1, Prof. Leandro Maciel Almeida).

## Contexto do projeto

- **Dataset**: Diabetes 130-US Hospitals 1999-2008 (UCI, ~101.766 instâncias, 50 features)
- **Tarefa**: classificação multiclasse, alvo `readmitted` ∈ {`<30`, `>30`, `NO`}
- **Métrica principal**: F1-macro (forte desbalanceamento; classe `<30` é minoritária)
- **Métricas secundárias**: balanced accuracy, ROC-AUC e PR macro multiclasse, análise por classe
- **Briefing**: hospital fictício quer prever readmissão precoce (<30 dias) para acionar acompanhamento pós-alta. Erros têm custos assimétricos (falso negativo na classe `<30` é mais grave).

## Pré-requisitos para nível Excelente (rubrica)

Três requisitos transversais — sem eles, **teto é nível Bom** em vários critérios:

1. **Reprodutibilidade end-to-end**: `main.ipynb` executa de ponta a ponta com no máximo um comando de instalação + um de execução. Não pode depender de arquivos manuais ou ordem mental.
2. **Comparação pareada**: toda variante vs baseline usa Wilcoxon ou paired t-test sobre métricas por fold, com p-valor reportado.
3. **Curvas treino vs validação**: para cada configuração de hiperparâmetro testada, em todos os experimentos.

## Cronograma e pesos da rubrica

**Datas críticas:**

| Data | Atividade |
|---|---|
| 20/05/2026 | Checkpoint do projeto (estado parcial dos experimentos) |
| 27/05/2026 | Apresentação dos resultados (15 min cronometrados) |
| 10/06/2026 | Entrega final do relatório + slides |

**Pesos da rubrica (priorizar esforço nesta ordem):**

| Peso | Critério |
|---|---|
| 18% | Busca de hiperparâmetros + comparação entre modelos |
| 15% | Modeling (10 algoritmos com configurações justificadas) |
| 13% | Justificativa experimental transversal |
| 12% | Data Preparation (baseline + variantes) |
| 10% | EDA |
| 8% | Business Understanding + briefing |
| 8% | Evaluation com CV no treino |
| 5% | Avaliação final no teste |
| 5% | Apresentação |
| 4% | Deployment |
| 2% | Integridade do split (peso pequeno, mas violar zera + penaliza Modeling e Avaliação) |

**Penalidades automáticas (da rubrica, "Notas operacionais"):**

- Faltar 2+ dos 10 algoritmos → rebaixa **um nível inteiro** em Modeling **e** Busca HP.
- Sem Wilcoxon/paired t-test + curvas treino vs validação → **teto Básico** nos critérios principais.
- Reprodutibilidade quebrada → **inelegível para Excelente** em qualquer critério que dependa de evidência experimental.

## Regra inegociável: integridade do split treino/teste

**O conjunto de teste é tocado UMA ÚNICA VEZ, na avaliação final.**

- Split estratificado logo no início, `random_state=42` fixo.
- EDA, pré-processamento, busca de hiperparâmetros, variantes, comparação — tudo só no treino com CV interna.
- A função `load_test(unlock_token)` em `src/data_loader.py` só libera o teste com o token `I_AM_IN_FINAL_EVALUATION`. Esse token aparece **apenas** na seção final do `main.ipynb`.
- Violar isso zera critérios da rubrica e penaliza Modeling. Nunca sugira "vamos dar uma olhada rápida no teste".

## Estrutura do repositório

```
.
├── CLAUDE.md                    # este arquivo
├── README.md                    # como rodar o projeto (comandos abaixo)
├── pyproject.toml               # uv — deps e config
├── uv.lock                      # lockfile reprodutível
├── main.ipynb                   # NOTEBOOK PRINCIPAL — executa ponta a ponta
├── data/
│   ├── raw/                     # CSV original do UCI (gitignored se >100MB)
│   ├── interim/                 # train/test split isolados após 1ª execução
│   └── processed/               # saídas dos pipelines de preparação
├── notebooks/                   # exploratórios opcionais, NÃO obrigatórios
│   ├── eda_exploration.ipynb    # rascunhos de EDA, plots descartados
│   └── hp_tuning_scratch.ipynb  # tentativas de espaço de busca
├── src/                         # código importado pelo main.ipynb
│   ├── data_loader.py           # load_train(), load_test(token) com guarda
│   ├── preprocessing.py         # pipelines sklearn (baseline + variantes)
│   ├── models.py                # factories dos 10 algoritmos
│   ├── search.py                # wrappers de Grid/RandomizedSearchCV
│   ├── evaluation.py            # métricas, testes pareados, plots
│   └── utils.py                 # logging, seeds, helpers
├── reports/
│   ├── figures/                 # PNGs gerados (curvas, matrizes, ROC, PR)
│   ├── cv_results/              # CSVs de cv_results_ por modelo
│   └── final/                   # relatorio.pdf, slides.pdf
└── tests/                       # smoke tests dos pipelines (opcional)
```

**Seções do `main.ipynb`** (na ordem do workflow obrigatório):
1. Setup e imports
2. Carga bruta + split estratificado treino/teste (teste isolado a partir daqui)
3. EDA somente no treino
4. Pipeline baseline (pré-processamento mínimo)
5. Busca de hiperparâmetros para os 10 algoritmos sobre o baseline (com curvas treino vs validação)
6. Variantes do treino (balanceamento, escala, codificação, FE) + nova busca HP em cada
7. Comparação baseline vs variantes com Wilcoxon/paired t-test
8. Seleção do melhor modelo justificada
9. **Avaliação final no teste** (única seção que usa o token de unlock)
10. Deployment, riscos, monitoramento

## Comandos (reprodução end-to-end)

Ambiente: Python + **uv** (https://docs.astral.sh/uv/).

```bash
# Setup (1 comando)
uv sync

# Reprodução completa (1 comando) — executa o main.ipynb de ponta a ponta
uv run jupyter nbconvert --to notebook --execute main.ipynb --output main.executed.ipynb

# Ou abrir interativamente
uv run jupyter lab main.ipynb

# Lint / format / testes
uv run ruff check src/ main.ipynb
uv run ruff format src/
uv run pytest tests/ -v
```

## Algoritmos obrigatórios (10 — mínimo absoluto)

Parte 1: **K-NN**, **LVQ** (via `sklvq`), **Árvore de Decisão**, **SVM**, **Random Forest**.
Parte 2: **MLP**, **Comitê de RNAs** (voting/bagging de MLPs), **Stacking heterogêneo**, **XGBoost**, **LightGBM**.

⚠️ **Faltar 2+ algoritmos rebaixa um nível inteiro** em Modeling e Busca HP (rubrica). Se algum falhar tecnicamente, documentar tentativa + falha em vez de simplesmente omitir.

Espaços de busca e estratégia (Grid vs Randomized) em `src/search.py`, seguindo a Tabela 1 das exigências. Sempre **5 folds** estratificados, **mesmos folds** entre experimentos (`random_state` fixo) — comparações pareadas exigem isso.

## Gráficos e entregáveis obrigatórios

**Briefing do cliente** (Seção 2.2 das exigências) — texto obrigatório no início do relatório. Modelo literal a preencher:

> O cliente *[nome fictício]* atua no setor *[setor]* e precisa de um modelo capaz de *[tarefa de classificação]*, em que a saída pertença a uma das classes *[<30, >30, NO]*. O sucesso será medido por *[F1-macro]* sobre o conjunto de teste, com restrição de *[tempo de inferência, custo de erro, etc.]*. As implicações éticas relevantes são *[viés, privacidade, impacto social]*.

**Variantes mínimas** (≥3 para nível Excelente em Data Preparation):

- Balanceamento: SMOTE, undersampling, `class_weight`
- Escalas: `StandardScaler`, `MinMaxScaler`, `RobustScaler`
- Codificações: OneHot, Target, Ordinal
- Engenharia de características baseada em evidência da EDA

**Gráficos exigidos** (Seção 9 das exigências):

- Curva **treino vs validação** para cada configuração de HP testada (todos os 10 modelos × todas as variantes).
- Matriz de confusão **normalizada** do melhor modelo de cada algoritmo, no baseline **e** na melhor variante.
- **Curvas ROC e PR macro multiclasse** dos modelos finais.
- Barras comparativas de **F1-macro entre baseline e cada variante**.

**Saídas da Seção 9 do `main.ipynb`** (avaliação final no teste): relatório de classificação + matriz de confusão normalizada + ROC macro + PR macro + análise por classe + comparação honesta com a CV.

**Formato de entrega final:**

- Relatório PDF baseado em `relatorio.tex`, **máximo 25 páginas** (sem apêndices).
- Slides PDF baseados em `slides.tex`.
- Logs (`cv_results_` em CSV ou runs MLflow) versionados ou linkados a partir do relatório.
- Repositório privado com o **professor adicionado como colaborador**.
- Seção final do relatório "Reprodutibilidade e ferramentas" **declarando uso de IA generativa** (exigido pela Seção 14 — integridade acadêmica).

**Apresentação (27/05/2026):** 15 min cronometrados (corta se exceder), **todos os integrantes apresentam** (presença verificada), sem código nos slides, ênfase em busca HP + comparação baseline vs variantes.

## Convenções

- **Sempre 5-fold estratificado**, `random_state=42`, mesmos folds entre experimentos.
- **Toda decisão técnica precisa de evidência experimental**: cite métrica com desvio entre folds (ex.: `F1-macro = 0.812 ± 0.014`).
- **Comparações baseline vs variante**: Wilcoxon pareado OU paired t-test sobre métricas por fold, reportar p-valor.
- **Curva treino vs validação** para cada configuração de hiperparâmetro testada, salva em `reports/figures/`.
- **Persistir `cv_results_`** como CSV em `reports/cv_results/` — o relatório precisa retomar valores específicos.
- **Pré-processamento dentro de `Pipeline` sklearn**, nunca aplicado ao dataset inteiro antes do split.
- **Categóricas do UCI Diabetes**: `race`, `gender`, `age` (buckets), `diag_1/2/3` (códigos ICD-9, alta cardinalidade — considerar agrupamento), 24 colunas de medicações. Faltantes marcados como `?`.

## Nunca faça

- ❌ Chamar `load_test()` sem o token `I_AM_IN_FINAL_EVALUATION` ou fora da seção 9 do `main.ipynb`.
- ❌ Aplicar `fit_transform` no dataset completo antes do split.
- ❌ Aplicar SMOTE ou qualquer reamostragem fora de um `Pipeline` de CV (vaza informação entre folds).
- ❌ Usar `accuracy_score` como métrica principal — dataset desbalanceado, exige F1-macro/balanced accuracy.
- ❌ Comparar modelos sem teste estatístico pareado quando for decidir variante vencedora.
- ❌ Rodar `n_jobs=-1` em buscas pesadas sem checar memória; usar 2-4 em máquinas modestas.
- ❌ Quebrar a reprodutibilidade do `main.ipynb` com células fora de ordem, paths absolutos ou estado oculto.
- ❌ Commitar `data/raw/` se exceder limite do GitHub — usar `.gitignore` e documentar download no README.

## Referências

- Exigências e rubrica: `docs/exigencias.pdf`, `docs/rubrica.pdf`
- Template do relatório: `docs/relatorio.pdf` (versão `.tex` precisa ser obtida com o professor — não está no repo)
- Template dos slides: `slides.tex` (também precisa ser obtido — não está no repo)
- Dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Paper original: Strack et al., 2014 (BioMed Research International)
