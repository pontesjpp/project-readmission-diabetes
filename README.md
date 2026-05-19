# Projeto Final IF1014 — Mineração de Dados (CRISP-DM)

Classificação multiclasse de readmissão hospitalar usando o dataset **Diabetes 130-US Hospitals** (UCI 296). Disciplina IF1014, UFPE 2026.1, Prof. Leandro Maciel Almeida.

Regras, rubrica e convenções inegociáveis estão em [CLAUDE.md](CLAUDE.md). Exigências e rubrica completas em [docs/](docs/).

## Reprodução

Pré-requisito: [uv](https://docs.astral.sh/uv/) instalado.

```bash
# 1. Instalar dependências (uma vez)
uv sync

# 2. Rodar o notebook principal ponta a ponta
uv run jupyter nbconvert --to notebook --execute main.ipynb --output main.executed.ipynb

# Ou abrir interativamente
uv run jupyter lab main.ipynb
```

Na primeira execução, `src/data_loader.py` baixa o dataset do UCI ML Repository via `ucimlrepo`, faz o split estratificado `train_test_split(test_size=0.2, random_state=42, stratify=y)` e salva em `data/interim/`. Execuções seguintes reutilizam o split persistido.

## Estrutura

- [main.ipynb](main.ipynb) — workflow obrigatório (10 seções na ordem do `docs/exigencias.pdf` §7).
- [src/](src/) — código importado pelo notebook. Nenhum estado oculto em variáveis globais.
- [data/interim/](data/interim/) — split treino/teste persistido (gerado, gitignored).
- [reports/figures/](reports/figures/) — curvas treino vs validação, matrizes de confusão, ROC/PR.
- [reports/cv_results/](reports/cv_results/) — `cv_results_` de cada busca em CSV.
- [docs/](docs/) — exigências, rubrica e material de referência do aluno.

## Regra inegociável: split treino/teste

O conjunto de teste é tocado **uma única vez**, na Seção 9 do `main.ipynb`. A função `load_test()` em [src/data_loader.py](src/data_loader.py) só libera o teste com o token literal `I_AM_IN_FINAL_EVALUATION`. Esse token aparece apenas na seção final do notebook.

Qualquer outra chamada (ex.: `load_test('peek')`) levanta `PermissionError` — não silencie esse erro.

## Comandos úteis

```bash
# Lint / format
uv run ruff check src/ main.ipynb
uv run ruff format src/
```
