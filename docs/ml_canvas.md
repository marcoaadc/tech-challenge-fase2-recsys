# ML Canvas — Sistema de Recomendação para E-commerce

## 1. Problema de Negócio

Um e-commerce quer aumentar a descoberta de produtos e a conversão. A página inicial e as vitrines exibem hoje apenas itens populares, o que concentra a atenção em uma fração pequena do catálogo e desperdiça o sinal mais abundante disponível: o **comportamento de navegação** dos usuários (cliques, visualizações, compras). O objetivo é recomendar, para cada usuário, os **top-K itens** com maior probabilidade de interação futura, personalizando a experiência e ampliando a exposição do catálogo.

## 2. Proposta de Valor

| Para quem | Valor entregue |
|-----------|----------------|
| Usuário final | Descobre produtos relevantes com menos esforço de busca |
| Negócio (comercial) | Aumento de CTR, conversão e ticket médio via personalização |
| Catálogo / sellers | Maior cobertura de itens expostos (long tail), não só os populares |
| Equipe de dados | Pipeline reprodutível (DVC + MLflow) que serve de base para iterações futuras |

## 3. Stakeholders

| Stakeholder | Papel | Interesse |
|-------------|-------|-----------|
| Diretoria de E-commerce | Patrocinador | Aumentar conversão e receita por sessão |
| Time de Produto | Usuário final | Vitrines personalizadas ("recomendados para você") |
| Equipe de Dados / ML | Desenvolvedor | Construir, avaliar e manter o modelo em produção |
| Sellers / gestão de catálogo | Consumidor indireto | Exposição mais equilibrada dos itens do catálogo |

## 4. Tarefa de ML

- **Tipo:** ranking top-K com **feedback implícito** (aprendizado de preferência a partir de interações, sem ratings explícitos).
- **Formulação:** classificação binária par (usuário, item) → probabilidade de interação; o logit é usado como score de ranqueamento.
- **Saída:** lista ordenada de K=10 itens por usuário, excluindo itens já consumidos.
- **Negativos:** amostragem negativa 4:1 sobre itens não interagidos (feedback implícito só possui positivos).

## 5. Dados

- **Fonte em produção (alvo):** eventos de navegação do e-commerce — cliques, visualizações de produto, adições ao carrinho e compras, com timestamp.
- **Fonte neste projeto (proxy):** **MovieLens 100K** (100 mil avaliações), com conversão para feedback implícito: rating ≥ 4 ⇒ interação positiva (`RatingThresholdStrategy`).
- **Filtros de qualidade:** mínimo de 5 interações por usuário e por item.
- **Volume após filtros:** **54.413 interações positivas**, **938 usuários**, **1.008 itens**.
- **Split:** temporal por usuário, **80/10/10** (treino/validação/teste) — treina no passado, avalia no futuro, sem vazamento temporal.
- **Limitação declarada:** filmes ≠ produtos; sazonalidade, recompra e carrinho não estão representados (ver [Model Card](model_card.md)).

## 6. Features

| Feature | Tipo | Uso |
|---------|------|-----|
| `user_id` (mapeado) | Categórica de alta cardinalidade | Embedding de usuário (dim 64) |
| `item_id` (mapeado) | Categórica de alta cardinalidade | Embedding de item (dim 64) |
| Histórico de interações | Implícito no treino | Define os pares positivos e o filtro de itens já vistos na inferência |
| Timestamp | Ordenação | Split temporal por usuário (não é feature do modelo) |

O modelo atual **não usa features de conteúdo** (categoria, preço, texto) — apenas IDs. Essa é a origem da limitação de cold-start (Seção 10).

## 7. Modelo

- **Arquitetura:** `EmbeddingMLP` (PyTorch) — embeddings de usuário e item (dim 64) concatenados, seguidos de MLP `[128, 64]` com dropout 0.2, produzindo um logit de interação.
- **Treino:** `BCEWithLogitsLoss` + Adam (lr=0.005, weight_decay=1e-4), batch 1024, early stopping (patience=3) monitorando recall@10 na validação.
- **Baselines:** Popularidade e ItemKNN — obrigatórios para provar que o modelo neural agrega valor.
- **Reprodutibilidade:** seeds determinísticos (42), pipeline DVC de 5 estágios, `poetry.lock` + `dvc.lock`.
- **Versionamento:** MLflow Model Registry — `recsys-embedding-mlp` v1 em **Production** (ciclo None → Staging → Production via `scripts/promote_model.py`).

## 8. Métricas

### Métricas Offline (conjunto de teste, protocolo full-ranking, K=10)

| Métrica | EmbeddingMLP | Justificativa |
|---------|--------------|---------------|
| Precision@10 | 0.0962 | Fração de recomendações relevantes no top-10 |
| Recall@10 | 0.1128 | Fração dos itens relevantes recuperados no top-10 (métrica de seleção do modelo) |
| NDCG@10 | 0.1255 | Qualidade da ordenação — relevantes mais alto na lista valem mais |
| HitRate@10 | 0.5288 | % de usuários com ao menos 1 acerto no top-10 (proxy de "vitrine útil") |
| AUC | 0.8280 | Capacidade global de ordenar positivos acima de negativos |

O EmbeddingMLP supera o baseline de popularidade em todas as métricas e o ItemKNN em precision, NDCG, hit rate e AUC (fonte: `reports/metrics.json`).

### Métricas Online (propostas, exigem deploy + experimentação)

| Métrica | Definição | Por que importa |
|---------|-----------|-----------------|
| CTR das recomendações | Cliques / impressões da vitrine recomendada | Proxy direto de relevância percebida |
| Taxa de conversão | Compras atribuídas às recomendações / sessões expostas | Impacto de negócio final |
| Cobertura de catálogo | % de itens distintos recomendados em uma janela | Detecta concentração excessiva em populares |
| Diversidade intra-lista | Dissimilaridade média entre itens do top-10 | Evita listas monotemáticas |

> Métricas offline são proxy imperfeito das online; a validação definitiva requer teste A/B (não realizado neste projeto).

## 9. Inferência

- **Modo:** batch de candidatos por usuário — o modelo pontua os itens do catálogo para cada usuário conhecido (full-ranking, excluindo itens já vistos) e materializa o top-10.
- **Cadência sugerida:** recomputo diário (ou por sessão, se houver serving em tempo real no futuro).
- **Escopo:** apenas usuários e itens **presentes no treino** (mapeamentos em `data/processed/mappings.json`); fora do vocabulário → fallback por popularidade.

## 10. Obstáculos e Riscos

| Risco | Descrição | Mitigação |
|-------|-----------|-----------|
| **Cold-start (usuário/item)** | IDs fora do vocabulário de treino não têm embedding | Fallback por popularidade; evolução futura com features de conteúdo |
| **Viés de popularidade** | Itens populares dominam os positivos e são super-recomendados, reduzindo a long tail | Monitorar cobertura de catálogo; re-ranking com diversidade |
| **Feedback loop** | Recomendações alimentam os dados do próximo treino, amplificando o viés a cada ciclo | Exploração controlada (ε de itens fora do top-K); auditoria de drift de popularidade |
| **Proxy de dados** | MovieLens não captura dinâmicas de e-commerce (sazonalidade, recompra) | Re-treino obrigatório em dados reais antes de uso comercial |
| **Negativos ruidosos** | Ausência de interação ≠ desinteresse; negativos amostrados contêm falsos negativos | Aceito como limitação do feedback implícito; documentado no Model Card |
| **Escala** | 938 usuários × 1.008 itens é muito menor que um catálogo real | Validar a vantagem do modelo neural vs. ItemKNN ao escalar |

## 11. Cenários de Falha

| Cenário | Impacto | Mitigação |
|---------|---------|-----------|
| Modelo recomenda só populares | Vitrine igual para todos; perda do valor da personalização | Alerta de cobertura de catálogo baixa (ver [Plano de Monitoramento](monitoring_plan.md)) |
| Scores degenerados (todos ≈ iguais) | Ranking aleatório disfarçado | Monitorar distribuição de scores; comparar contra baseline |
| % alto de usuários cold-start | Grande parte dos usuários sem personalização | Monitorar % de fallback; priorizar estratégia de cold-start |
| Drift silencioso pós-deploy | CTR/conversão caem sem erro técnico | Alertas de negócio + re-treino via `dvc repro` |
