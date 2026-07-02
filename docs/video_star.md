# Roteiro do vídeo (5 min) — Método STAR

Apresentação do Tech Challenge FIAP Fase 02: sistema de recomendação para e-commerce com PyTorch, DVC, MLflow e Docker.

**Preparação antes de gravar:**

- Terminal aberto na raiz do repositório, com o ambiente Poetry ativo.
- MLflow UI rodando (`make mlflow-ui`) com o experimento `recsys-ecommerce` aberto.
- `docker compose up mlflow` testado previamente (opcional, para o bloco Action).
- Editor com os arquivos-chave em abas: `dvc.yaml`, `configs/params.yaml`, `src/recsys/models/factory.py`, `src/recsys/data/preprocess.py`, `reports/metrics.json`.

---

## Situation — o problema de negócio (0:00 – 1:00)

**Falar:**

- Em um e-commerce, a maior parte do sinal de preferência do usuário é **implícita**: cliques, visualizações, compras — não avaliações explícitas.
- Objetivo: dado o histórico de interações user-item, **recomendar os top-10 itens** com maior probabilidade de interação para cada usuário.
- Dataset: **MovieLens 100K** como proxy público e reprodutível de interações user-item — 100 mil avaliações convertidas em feedback implícito (rating ≥ 4 = interação positiva).
- Após filtros de qualidade (≥ 5 interações por usuário e por item): **54.413 interações, 938 usuários, 1.008 itens**.
- Deixar explícita a limitação: filmes são proxy, não e-commerce real — documentado no Model Card.

**Mostrar na tela:**

- README §1 (problema) e a tabela do dataset.
- Rapidamente `data/processed/interactions.parquet` no explorador de arquivos ou o trecho de `preprocess` que aplica o threshold de rating.

## Task — objetivos técnicos e restrições (1:00 – 2:00)

**Falar:**

- O challenge pede um projeto de ML **ponta a ponta com práticas de engenharia**, não só um notebook. Metas definidas:
  1. Modelo de deep learning em **PyTorch** que supere baselines honestos (popularidade e ItemKNN).
  2. Pipeline **100% reprodutível**: DVC para dados e estágios, seeds determinísticos, Poetry com lock.
  3. **Experimentos rastreados** no MLflow e modelo versionado no Model Registry com ciclo de promoção.
  4. **Containerização** com Docker para eliminar "funciona na minha máquina".
  5. Qualidade de código: testes (pytest), lint (ruff), pre-commit, configuração via pydantic-settings.
- Restrição metodológica importante: **avaliação sem vazamento temporal** — split temporal por usuário (80/10/10) e ranking excluindo itens já vistos.

**Mostrar na tela:**

- Estrutura de pastas do repo (árvore no README §3).
- `Makefile` (os alvos resumem os requisitos: repro, test, lint, mlflow-ui, docker-up).

## Action — arquitetura e decisões (2:00 – 4:00)

**Falar (ordem sugerida):**

1. **Pipeline DVC de 5 estágios** — download → preprocess → build_features → train → evaluate. Cada estágio declara deps, params e outs; `dvc repro` só reexecuta o que mudou. `configs/params.yaml` é a fonte única de hiperparâmetros.
2. **Design patterns:**
   - **Strategy** no preprocessamento: `ImplicitFeedbackStrategy` torna intercambiável a regra de conversão rating → feedback implícito (threshold ≥ 4 vs todas as interações).
   - **Factory** nos modelos: `ModelFactory` cria o modelo a partir do `model_type` do params.yaml — trocar de arquitetura não toca no trainer.
3. **Modelo:** `EmbeddingMLP` — embeddings de usuário e item (dim 64) concatenados → MLP [128, 64] com dropout 0.2 → logit; `BCEWithLogitsLoss` + Adam; **negative sampling 4:1** (feedback implícito só tem positivos); **early stopping** com patience=3 monitorando recall@10 de validação.
4. **MLflow:** 4 runs de treino no experimento `recsys-ecommerce` — busca de hiperparâmetros (dim 32/lr 0.001 → 0.0781; dim 16/lr 0.005 → 0.1511; dim 64/lr 0.005 → **0.1541**, config final) e um run final de reprodução que bateu **exatamente** 0.1541 (seeds determinísticos). Melhor run registrado como `recsys-embedding-mlp` v1 e promovido **None → Staging → Production** via `scripts/promote_model.py`.
5. **Docker:** Dockerfile multi-stage (builder com Poetry, runtime slim não-root) e docker-compose com serviços `mlflow` (tracking server, porta 5000) e `train` (pipeline completo apontando para o MLflow).

**Mostrar na tela:**

- Terminal: `poetry run dvc dag` (grafo dos 5 estágios) e/ou trecho do `dvc.yaml`.
- Código: `src/recsys/data/preprocess.py` (classes Strategy) e `src/recsys/models/factory.py` — 10–15 segundos cada, apontando o padrão.
- **MLflow UI:** tabela de runs ordenada por `best_val_recall` (comparar os 4 runs) e a página do Model Registry com `recsys-embedding-mlp` em Production.
- `docker-compose.yml` no editor ou `docker compose up mlflow` rodando + UI no navegador.

## Result — métricas, trade-offs e lições (4:00 – 5:00)

**Falar:**

- Métricas finais no **teste** (full-ranking, K=10):

  | Modelo | P@10 | R@10 | NDCG@10 | HitRate@10 | AUC |
  |---|---|---|---|---|---|
  | EmbeddingMLP | **0.0962** | 0.1128 | **0.1255** | **0.5288** | **0.8280** |
  | ItemKNN | 0.0881 | 0.1155 | 0.1248 | 0.5160 | 0.7775 |
  | Popularidade | 0.0608 | 0.0671 | 0.0832 | 0.3849 | 0.7383 |

- **Leitura honesta (trade-off):** o modelo neural vence a popularidade em tudo e supera o ItemKNN em precision, NDCG, hit rate e principalmente AUC (+0.05) — mas o ItemKNN empata em recall@10. Em datasets pequenos, baselines fortes são competitivos; a comparação justa é parte do resultado.
- **Lições aprendidas:** (1) reprodutibilidade é decisão de arquitetura, não detalhe — seeds + DVC + lock permitiram reproduzir o recall exato; (2) split temporal e exclusão de itens vistos mudam muito as métricas — protocolo de avaliação importa tanto quanto o modelo; (3) baselines primeiro: sem eles, 0.1255 de NDCG não significa nada.
- Fechar: próximo passo natural seria servir o modelo de Production via API e avaliar online (A/B).

**Mostrar na tela:**

- `reports/metrics.json` ou a tabela de métricas do README §10.
- Encerrar com o `README.md` renderizado no GitHub (visão geral do projeto).

---

**Dicas de gravação:** ensaie o timing por bloco (1/1/2/1 min); deixe comandos pré-digitados no histórico do terminal; grave em 1080p com zoom no código; fale sobre *por que* das decisões, não leia o código linha a linha.
