# Tech Challenge FIAP Fase 02 — Sistema de Recomendação para E-commerce

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EmbeddingMLP-EE4C2C.svg)
![DVC](https://img.shields.io/badge/pipeline-DVC-945DD6.svg)
![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2.svg)
![Docker](https://img.shields.io/badge/container-Docker-2496ED.svg)

Sistema de recomendação de produtos para e-commerce baseado no comportamento de navegação dos usuários. O modelo é uma rede neural em PyTorch (embeddings de usuário e item + MLP), treinada com feedback implícito, dentro de um pipeline totalmente reprodutível (DVC), com experimentos rastreados e modelo versionado no MLflow (Tracking + Model Registry) e execução containerizada via Docker.

Repositório: <https://github.com/marcoaadc/tech-challenge-fase2-recsys>

---

## 1. O problema

Em um e-commerce, a maior parte do sinal sobre a preferência do usuário não vem de avaliações explícitas, e sim do **comportamento de navegação**: cliques, visualizações e compras. O objetivo aqui é, dado o histórico de interações user-item, **ranquear os itens que cada usuário tem maior probabilidade de consumir a seguir** (top-K recomendações).

Como proxy público e reprodutível de interações user-item, usamos o **MovieLens 100K** (100 mil avaliações). As avaliações são convertidas em **feedback implícito**: rating ≥ 4 vira interação positiva (equivalente a "clicou/comprou"). Após os filtros de qualidade (mínimo de 5 interações por usuário e por item), o dataset final tem:

| | Valor |
|---|---|
| Interações positivas | **54.413** |
| Usuários | **938** |
| Itens | **1.008** |

> As limitações dessa aproximação (filmes ≠ produtos) estão documentadas no [Model Card](docs/model_card.md).

## 2. Arquitetura da solução

Pipeline de 5 estágios orquestrado pelo DVC, com tracking de experimentos no MLflow:

```
                        configs/params.yaml (fonte única de hiperparâmetros)
                                        │
  ┌──────────┐   ┌────────────┐   ┌────────────────┐   ┌─────────┐   ┌──────────┐
  │ download │──▶│ preprocess │──▶│ build_features │──▶│  train  │──▶│ evaluate │
  └──────────┘   └────────────┘   └────────────────┘   └─────────┘   └──────────┘
       │               │                  │                 │              │
   data/raw/      data/processed/    data/features/    models/       reports/
   ml-100k        interactions      train/val/test    model.pt      metrics.json
                  + mappings        .parquet              │              │
                                                          ▼              ▼
                                                   ┌─────────────────────────┐
                                                   │      MLflow Tracking     │
                                                   │  params, métricas, runs  │
                                                   │            +             │
                                                   │      Model Registry      │
                                                   │  recsys-embedding-mlp v1 │
                                                   │  None→Staging→Production │
                                                   └─────────────────────────┘
```

**O modelo (`EmbeddingMLP`):** embeddings de usuário e item são concatenados e passam por um MLP que produz um logit de interação. Treino com `BCEWithLogitsLoss` + Adam, amostragem negativa 4:1 e early stopping (patience=3) monitorando recall@10 na validação.

## 3. Estrutura do repositório

```
techallenger2/
├── configs/
│   └── params.yaml            # Hiperparâmetros e paths (fonte única, versionada pelo DVC)
├── data/                      # Gerenciado pelo DVC (raw → processed → features)
├── docs/
│   ├── model_card.md          # Model Card do modelo em produção
│   ├── ml_canvas.md           # ML Canvas: enquadramento de negócio da solução
│   └── monitoring_plan.md     # Plano de monitoramento e playbooks de resposta
├── models/                    # model.pt (saída do stage train)
├── reports/                   # metrics.json (saída do stage evaluate)
├── scripts/
│   ├── promote_model.py       # Registra e promove o modelo no Model Registry
│   └── validate_env.py        # Valida o ambiente local
├── src/recsys/
│   ├── config/                # Settings (pydantic-settings + .env)
│   ├── data/                  # download + preprocess (Strategy p/ feedback implícito)
│   ├── features/              # split temporal + negative sampling
│   ├── models/                # EmbeddingMLP, baselines, ModelFactory, persistence
│   ├── training/              # Dataset e Trainer (early stopping, MLflow logging)
│   ├── evaluation/            # métricas de ranking (full-ranking) e evaluator
│   ├── pipelines/             # CLIs dos 5 estágios (1 módulo por stage DVC)
│   └── utils/                 # seeds determinísticos
├── tests/                     # 22 testes (pytest)
├── dvc.yaml / dvc.lock        # Definição do pipeline DVC
├── Dockerfile                 # Multi-stage (builder/runtime), usuário não-root
├── docker-compose.yml         # Serviços: mlflow (UI :5000) + train
├── Makefile                   # Atalhos: install, test, lint, repro, mlflow-ui...
└── pyproject.toml / poetry.lock
```

## 4. Requisitos

- Python **3.10+**
- [Poetry](https://python-poetry.org/) (gerenciamento de dependências; lock commitado)
- `make` (opcional, atalhos) — no Windows, use Git Bash
- Docker + Docker Compose (opcional, para execução containerizada)

## 5. Instalação do zero

```bash
git clone https://github.com/marcoaadc/tech-challenge-fase2-recsys.git
cd tech-challenge-fase2-recsys

# 1. Dependências (cria o virtualenv e instala do poetry.lock)
poetry install          # ou: make install

# 2. Configuração via variáveis de ambiente (pydantic-settings)
cp .env.example .env    # defaults sensatos; ajuste se necessário

# 3. (Opcional) validar o ambiente
make validate           # ou: poetry run python scripts/validate_env.py
```

## 6. Reproduzindo o pipeline

O pipeline inteiro (download → preprocess → build_features → train → evaluate) é reproduzível com um comando:

```bash
make repro              # ou: poetry run dvc repro
```

O DVC só reexecuta os estágios cujas dependências ou parâmetros mudaram. Alternativa sem DVC (mesmos artefatos, útil em ambientes sem git/dvc):

```bash
make pipeline           # encadeia os 5 CLIs: python -m recsys.pipelines.<stage>
```

Os dados versionados usam um **remote DVC local** em `../dvc-storage-recsys` (`dvc push` / `dvc pull`).

## 7. Experimentos (MLflow Tracking)

Cada treino loga parâmetros, métricas por época e artefatos no experimento **`recsys-ecommerce`**:

```bash
make mlflow-ui          # UI em http://localhost:5000
```

Busca de hiperparâmetros realizada (4 runs de treino, métrica: recall@10 na validação):

| Run | embedding_dim | learning_rate | hidden_dims | recall@10 (val) |
|---|---|---|---|---|
| 1 | 32 | 0.001 | — | 0.0781 |
| 2 | 16 | 0.005 | — | 0.1511 |
| 3 | **64** | **0.005** | **[128, 64]** | **0.1541** ← config final |
| 4 (repro) | 64 | 0.005 | [128, 64] | 0.1541 (reprodução exata) |

O run 4 reproduziu o resultado do run 3 **exatamente** (0.1541), graças aos seeds determinísticos (`seed=42` propagado para Python/NumPy/PyTorch).

## 8. Model Registry (registro e promoção)

O melhor run é registrado no MLflow Model Registry como **`recsys-embedding-mlp`** e promovido pelo ciclo de vida completo **None → Staging → Production**:

```bash
poetry run python scripts/promote_model.py
```

O script busca o run com melhor `best_val_recall` no experimento, registra a versão (v1) e a promove até `Production`.

## 9. Execução com Docker

```bash
docker compose up --build        # ou: make docker-up
```

Serviços definidos no `docker-compose.yml`:

| Serviço | O que faz |
|---|---|
| `mlflow` | Tracking server (SQLite + artifacts em volume), UI em <http://localhost:5000> |
| `train` | Builda a imagem (Dockerfile multi-stage, usuário não-root) e roda os 5 estágios do pipeline apontando para o MLflow |

Também é possível subir só o tracking server (`docker compose up mlflow`) ou buildar a imagem isoladamente (`make docker-build`).

## 10. Resultados

Avaliação **offline no conjunto de teste**, protocolo *full-ranking* (o modelo ranqueia todos os itens do catálogo, excluindo os já vistos pelo usuário), K=10. Fonte: `reports/metrics.json`.

| Modelo | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | AUC |
|---|---|---|---|---|---|
| **EmbeddingMLP (final)** | **0.0962** | 0.1128 | **0.1255** | **0.5288** | **0.8280** |
| ItemKNN (baseline) | 0.0881 | **0.1155** | 0.1248 | 0.5160 | 0.7775 |
| Popularidade (baseline) | 0.0608 | 0.0671 | 0.0832 | 0.3849 | 0.7383 |

**Leitura dos resultados:** o EmbeddingMLP supera claramente o baseline de popularidade em todas as métricas e supera o ItemKNN em precision, NDCG, hit rate e, com folga, em AUC (0.828 vs 0.778) — indicando ordenação global melhor. O ItemKNN permanece competitivo em recall@10, o que é esperado em datasets pequenos, e valida a importância de manter baselines fortes na comparação.

## 11. Decisões de design

| Decisão | Motivação |
|---|---|
| **Factory pattern** (`ModelFactory`, `src/recsys/models/factory.py`) | Modelos são criados por nome de tipo + hiperparâmetros (`model_type: embedding_mlp` no params.yaml), desacoplando o pipeline de treino das classes concretas. Adicionar uma arquitetura nova = registrar um builder, sem tocar no trainer. |
| **Strategy pattern** (`ImplicitFeedbackStrategy`, `src/recsys/data/preprocess.py`) | A regra que converte ratings em feedback implícito é intercambiável (`RatingThresholdStrategy` com rating ≥ 4 vs `AllInteractionsStrategy`), isolando uma decisão de negócio do código do pipeline. |
| **Split temporal por usuário (80/10/10)** | Em recomendação, split aleatório vaza o futuro para o treino. Ordenar as interações de cada usuário no tempo e cortar 80/10/10 simula o cenário real: treinar no passado, prever o futuro. |
| **Amostragem negativa 4:1** | Feedback implícito só tem positivos. Para cada interação positiva, 4 itens não interagidos são amostrados como negativos, viabilizando a classificação binária (BCE) sem explodir o custo de treino. |
| **Early stopping (patience=3)** | Monitora recall@10 na validação e interrompe o treino quando parar de melhorar — controla overfitting e evita desperdício de épocas. |
| **Seeds determinísticos** | `seed=42` fixado em Python/NumPy/PyTorch (`src/recsys/utils/seed.py`); a reprodução do run final bateu o recall@10 de validação exato (0.1541). |
| **Config única versionada** | `configs/params.yaml` é a fonte de todos os hiperparâmetros e é dependência declarada dos stages DVC — mudar um parâmetro invalida (e reexecuta) apenas os stages afetados. |
| **pydantic-settings + `.env`** | Configuração de ambiente (tracking URI, paths, seed) tipada e validada, com defaults sensatos. |

## 12. Qualidade: testes e lint

```bash
make test     # pytest — 22 testes (preprocess, features, factory, baselines, embedding_mlp, métricas)
make lint     # ruff check .
make format   # ruff format + fixes automáticos
```

Hooks de **pre-commit** configurados (`.pre-commit-config.yaml`) rodam o ruff antes de cada commit.

## 13. Mapeamento dos requisitos do Tech Challenge

| Requisito do challenge | Onde está atendido |
|---|---|
| Problema de ML relevante ponta a ponta | Recomendação top-K com feedback implícito (este README, §1–§2) |
| Coleta/preparação de dados | Stages `download` e `preprocess` (`src/recsys/data/`, `dvc.yaml`) |
| Engenharia de features | `src/recsys/features/build_features.py` (split temporal + negative sampling) |
| Modelo de deep learning (PyTorch) | `src/recsys/models/embedding_mlp.py` + `src/recsys/training/trainer.py` |
| Baselines para comparação | `src/recsys/models/baselines.py` (Popularidade, ItemKNN) |
| Pipeline reprodutível / versionamento de dados | **DVC**: `dvc.yaml` (5 stages), `dvc.lock`, remote local; `make repro` |
| Rastreamento de experimentos | **MLflow Tracking**: experimento `recsys-ecommerce`, 4 runs comparados (§7) |
| Registro/versionamento de modelo | **MLflow Model Registry**: `recsys-embedding-mlp` v1, None→Staging→Production (`scripts/promote_model.py`) |
| Avaliação com métricas adequadas | `src/recsys/evaluation/` — precision/recall/NDCG/hit rate@10 + AUC, full-ranking (§10) |
| Containerização | `Dockerfile` multi-stage não-root + `docker-compose.yml` (mlflow + train) |
| Gestão de dependências | Poetry com `poetry.lock` commitado |
| Configuração e segredos | `pydantic-settings` + `.env` (`.env.example` no repo) |
| Qualidade de código | `ruff` + `pre-commit` + `pytest` (22 testes) |
| Documentação | Este README, [Model Card](docs/model_card.md), [ML Canvas](docs/ml_canvas.md), [Plano de Monitoramento](docs/monitoring_plan.md) |

## 14. Documentação adicional

- [Model Card](docs/model_card.md) — detalhes do modelo, uso pretendido, limitações e considerações éticas.
- [ML Canvas](docs/ml_canvas.md) — enquadramento de negócio: problema, proposta de valor, dados, métricas e riscos.
- [Plano de Monitoramento](docs/monitoring_plan.md) — métricas em produção, alertas, detecção de drift e playbooks de resposta.

---

Projeto desenvolvido para o **Tech Challenge — FIAP, Fase 02** (Machine Learning Engineering).
