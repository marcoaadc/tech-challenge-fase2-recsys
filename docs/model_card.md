# Model Card — recsys-embedding-mlp

Model card do modelo de recomendação desenvolvido para o Tech Challenge FIAP Fase 02.

## Detalhes do modelo

| Campo | Valor |
|---|---|
| Nome no Model Registry | `recsys-embedding-mlp` |
| Versão | v1 (estágio **Production**; ciclo None → Staging → Production via `scripts/promote_model.py`) |
| Tipo | Rede neural de recomendação com feedback implícito (`EmbeddingMLP`) |
| Framework | PyTorch |
| Experimento MLflow | `recsys-ecommerce` |
| Código | `src/recsys/models/embedding_mlp.py` (instanciado via `ModelFactory`) |
| Artefato | `models/model.pt` (saída do stage `train` do DVC) |

### Arquitetura

Embeddings de usuário e item (dim 64 cada) são **concatenados** e passam por um MLP (camadas ocultas `[128, 64]`, dropout 0.2) que produz um **logit** de probabilidade de interação. O ranqueamento top-K usa o logit como score.

### Hiperparâmetros (config final, `configs/params.yaml`)

| Hiperparâmetro | Valor |
|---|---|
| `embedding_dim` | 64 |
| `hidden_dims` | [128, 64] |
| `dropout` | 0.2 |
| Loss | `BCEWithLogitsLoss` |
| Otimizador | Adam (`lr=0.005`, `weight_decay=1e-4`) |
| `batch_size` | 1024 |
| Early stopping | patience=3, monitorando recall@10 na validação |
| Negative sampling | 4 negativos por positivo |
| `seed` | 42 (determinístico em Python/NumPy/PyTorch) |

## Uso pretendido

- **Caso de uso primário:** ranquear itens do catálogo para cada usuário conhecido (recomendação top-10), a partir do histórico de interações implícitas, em cenário de e-commerce.
- **Usuários-alvo:** avaliadores do Tech Challenge e estudo de arquitetura MLOps (DVC + MLflow + Docker) para sistemas de recomendação.
- **Escopo de dados:** usuários e itens **presentes no treino** (IDs mapeados em `data/processed/mappings.json`).

### Fora de escopo

- **Produção comercial real** sem re-treino em dados do domínio: o modelo foi treinado em um proxy (filmes), não em navegação de e-commerce real.
- **Usuários ou itens novos (cold-start):** o modelo não possui features de conteúdo; IDs fora do vocabulário de treino não têm embedding.
- **Decisões sensíveis sobre pessoas** (crédito, preços personalizados, elegibilidade): o modelo prevê apenas afinidade de consumo.
- **Garantias online:** nenhuma avaliação online (A/B test) foi realizada.

## Dados de treino

- **Origem:** [MovieLens 100K](https://files.grouplens.org/datasets/movielens/ml-100k.zip) (GroupLens) — 100.000 avaliações explícitas, usado como **proxy** de interações user-item de e-commerce.
- **Conversão para feedback implícito:** rating ≥ 4 ⇒ interação positiva (`RatingThresholdStrategy`, padrão Strategy em `src/recsys/data/preprocess.py`).
- **Filtros:** mínimo de 5 interações por usuário e por item.
- **Resultado após filtros:** **54.413 interações**, **938 usuários**, **1.008 itens**.
- **Split:** **temporal por usuário, 80/10/10** (treino/validação/teste) — as interações mais recentes de cada usuário ficam na validação e no teste, evitando vazamento do futuro.
- **Negativos:** amostragem 4:1 sobre itens não interagidos (para treino/validação).

## Métricas de avaliação

Avaliação offline no conjunto de **teste**, protocolo *full-ranking* (score de todos os itens do catálogo por usuário, excluindo itens já vistos), K=10. Fonte: `reports/metrics.json`.

| Modelo | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | AUC |
|---|---|---|---|---|---|
| **embedding_mlp (este modelo)** | **0.0962** | 0.1128 | **0.1255** | **0.5288** | **0.8280** |
| item_knn (baseline) | 0.0881 | **0.1155** | 0.1248 | 0.5160 | 0.7775 |
| popularity (baseline) | 0.0608 | 0.0671 | 0.0832 | 0.3849 | 0.7383 |

Seleção de modelo feita por recall@10 na **validação** (melhor run: 0.1541, reproduzido exatamente em um segundo run com os mesmos seeds).

## Limitações

1. **Dataset proxy:** MovieLens é consumo de filmes, não navegação de e-commerce. Dinâmicas como sazonalidade de compra, recompra e carrinho não estão representadas; as métricas não se transferem diretamente para um catálogo real.
2. **Cold-start:** usuários e itens não vistos no treino não são atendidos (embeddings apenas por ID, sem features de conteúdo).
3. **Viés de popularidade:** o feedback implícito reflete o que foi exposto e consumido; itens populares dominam os positivos e tendem a ser super-recomendados, reduzindo cobertura de catálogo (long tail).
4. **Escala pequena:** 938 usuários × 1.008 itens é ordens de magnitude menor que um e-commerce real; conclusões sobre a vantagem do modelo neural sobre o ItemKNN podem não escalar.
5. **Avaliação offline apenas:** métricas de ranking offline são proxy imperfeito de impacto de negócio (CTR, conversão, receita). Não houve teste online.
6. **Rótulos implícitos ruidosos:** ausência de interação não significa desinteresse — os negativos amostrados podem conter falsos negativos.

## Vieses e considerações éticas

- **Feedback loop de popularidade:** se as recomendações do modelo alimentarem os dados do próximo treino, itens populares recebem ainda mais exposição, amplificando o viés a cada ciclo. Mitigações possíveis: re-ranking com diversidade, exploração controlada e monitoramento de cobertura de catálogo.
- **Viés de exposição:** os usuários só interagem com o que lhes foi mostrado; o dataset não registra o que nunca foi exposto. O modelo aprende a preferência *condicionada à exposição histórica*, não a preferência real.
- **Grupos sub-representados:** usuários com pouco histórico (ou nichos de itens) recebem recomendações de pior qualidade, o que pode degradar a experiência exatamente de quem tem menos dados.
- **Privacidade:** o modelo usa apenas IDs anônimos e timestamps; nenhum dado pessoal identificável é consumido. Em produção real, o histórico de navegação exigiria base legal e políticas de retenção adequadas (LGPD).

## Manutenção e re-treino

- **Re-treino completo:** `dvc repro` (ou `make repro`) reexecuta os 5 estágios (download → preprocess → build_features → train → evaluate) apenas onde houver mudança de código, dados ou parâmetros. Hiperparâmetros são editados em `configs/params.yaml`.
- **Tracking:** todo treino gera um run no experimento MLflow `recsys-ecommerce` (`make mlflow-ui` para inspecionar).
- **Promoção:** após validar as métricas do novo run, `poetry run python scripts/promote_model.py` registra a nova versão no Registry e a promove (None → Staging → Production).
- **Reprodutibilidade:** seeds fixos (42) + `poetry.lock` + `dvc.lock` garantem reprodução bit a bit do resultado reportado **na mesma plataforma que o gerou** (Windows/CPU, torch 2.12.1). O container Linux usa torch CPU-only (`2.13.0+cpu`); entre plataformas espera-se equivalência estatística, não igualdade exata.
- **Contato:** Marco (marcoaadc@gmail.com) — Tech Challenge FIAP Fase 02.
