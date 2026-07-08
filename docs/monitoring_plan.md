# Plano de Monitoramento — Sistema de Recomendação (recsys-embedding-mlp)

Plano de monitoramento em produção para o modelo `recsys-embedding-mlp` (MLflow Model Registry, v1 em **Production**). Assume serving do top-10 por usuário via batch diário de scoring (full-ranking com exclusão de itens já vistos), com evolução possível para serving online.

## 1. Métricas Monitoradas

### Métricas Técnicas (serving/pipeline)

| Métrica | Descrição | Fonte | Frequência |
|---------|-----------|-------|------------|
| **Latência de scoring** | Tempo do job batch de scoring por usuário (p50/p95) e duração total do job | Logs do job de inferência | Por execução |
| **Taxa de erro** | % de usuários cujo scoring falhou (ID sem mapping, exceção de inferência) | Logs do job | Por execução |
| **Frescor das recomendações** | Idade da última lista top-10 materializada | Timestamp do artefato de saída | Diária |
| **Sucesso do pipeline** | Status dos 5 estágios DVC (download → evaluate) no re-treino | `dvc repro` / CI | Por re-treino |
| **Uso de memória/CPU** | Recursos do container de inferência (Docker) | Runtime metrics | Contínuo |

### Métricas de Modelo

| Métrica | Descrição | Fonte | Frequência |
|---------|-----------|-------|------------|
| **Distribuição de scores** | Histograma dos logits/probabilidades do top-10 servido | Logs de scoring | Diária |
| **Drift de popularidade** | Concentração das recomendações nos itens mais populares (ex.: % do top-10 vindo do top-5% de itens por popularidade) vs. baseline do deploy | Listas top-10 materializadas | Semanal |
| **Cobertura de catálogo** | % de itens distintos do catálogo que aparecem em algum top-10 na janela de 7 dias | Listas top-10 materializadas | Semanal |
| **% de usuários cold-start** | % de usuários ativos sem recomendação personalizada (fora do vocabulário → fallback por popularidade) | Logs de scoring + `mappings.json` | Diária |
| **Métricas offline recorrentes** | Precision/Recall/NDCG/HitRate@10 e AUC no teste, a cada re-treino | `reports/metrics.json` (stage `evaluate`) | Por re-treino |

### Métricas de Negócio

| Métrica | Descrição | Fonte | Frequência |
|---------|-----------|-------|------------|
| **CTR das recomendações** | Cliques / impressões das vitrines recomendadas | Eventos de front-end | Diária |
| **Conversão atribuída** | Compras originadas de cliques em recomendação / sessões expostas | Eventos + atribuição | Semanal |
| **Diversidade intra-lista** | Dissimilaridade média entre os 10 itens recomendados por usuário | Listas top-10 | Semanal |
| **Uplift vs. popularidade** | CTR do modelo vs. CTR de vitrine de populares (grupo de controle) | Experimento A/B contínuo (holdout pequeno) | Semanal |

## 2. Alertas e Thresholds

| Alerta | Condição | Severidade | Ação |
|--------|----------|------------|------|
| **Job de scoring falhou** | Batch diário não concluiu ou taxa de erro > 5% dos usuários | Critical | Investigar logs; se artefato `model.pt` corrompido, restaurar do Registry |
| **Latência de scoring alta** | Duração do job > 2× a média histórica | Warning | Verificar recursos do container; investigar crescimento do catálogo |
| **Recomendações velhas** | Última lista top-10 com > 48h | Critical | Reexecutar job; comunicar produto (vitrine desatualizada) |
| **Queda de CTR** | CTR médio 7 dias < 85% do baseline dos 28 dias anteriores (queda > 15%) | Warning | Seguir playbook "Queda de CTR" (Seção 4) |
| **Queda de CTR crítica** | CTR médio 7 dias < 70% do baseline (queda > 30%) | Critical | Playbook "Queda de CTR" + considerar rollback imediato |
| **Cobertura de catálogo baixa** | < 20% dos itens aparecem em algum top-10 na semana | Warning | Investigar viés de popularidade; avaliar re-ranking com diversidade |
| **Drift de popularidade** | % do top-10 vindo do top-5% de itens sobe > 10 p.p. vs. baseline do deploy | Warning | Indício de feedback loop; seguir playbook "Drift" |
| **Cold-start alto** | > 15% dos usuários ativos servidos por fallback de popularidade | Warning | Antecipar re-treino para incorporar novos usuários/itens ao vocabulário |
| **Scores degenerados** | Desvio-padrão dos scores do top-10 cai > 50% vs. baseline | Warning | Verificar integridade do modelo; comparar contra run registrado no MLflow |
| **Métrica offline degradada** | Recall@10 do novo treino < 0.10 ou AUC < 0.78 (piso ≈ ItemKNN 0.7775) | Critical | Não promover a nova versão; investigar dados/params |

> Baselines de referência do modelo em produção (v1): Precision@10 0.0962, Recall@10 0.1128, NDCG@10 0.1255, HitRate@10 0.5288, AUC 0.8280 (`reports/metrics.json`).

## 3. Detecção de Drift (específico de recsys)

| Método | Aplicação | Threshold |
|--------|-----------|-----------|
| **PSI sobre distribuição de scores** | Scores do top-10 servido: janela atual vs. janela do deploy | PSI > 0.2 → drift significativo |
| **Distribuição de popularidade dos itens recomendados** | Comparar histograma de rank de popularidade dos itens no top-10 vs. baseline | Deslocamento > 10 p.p. na massa do top-5% → investigar |
| **Cobertura de catálogo (tendência)** | Série semanal da cobertura | Queda por 3 semanas consecutivas → investigar |
| **Volume de interações por usuário** | Mudança na distribuição de atividade (entrada do modelo no próximo treino) | KS p-value < 0.05 → drift de comportamento |
| **Taxa de novos usuários/itens** | % de IDs fora do vocabulário de treino | > 15% → re-treino antecipado |

**Atenção ao feedback loop:** em recomendação, o próprio modelo altera os dados futuros (usuários clicam no que foi mostrado). Drift de popularidade crescente e cobertura decrescente juntos são a assinatura clássica desse loop — a resposta não é só re-treinar, mas adicionar exploração/diversidade no re-ranking.

## 4. Playbooks de Resposta

### Incidente: Queda de CTR > 15% em 7 dias

```
1. Confirmar que a queda não é sazonal/externa (campanha encerrada, tráfego pago,
   mudança de layout da vitrine) — comparar com o grupo de controle (populares).
2. Verificar frescor: listas top-10 estão sendo atualizadas? (alerta de 48h)
3. Verificar % de cold-start: subiu? Muitos usuários caindo no fallback derruba o CTR médio.
4. Calcular PSI dos scores e drift de popularidade (Seção 3).
5. Se drift confirmado: re-treinar com dados recentes → `dvc repro` (ou `make repro`).
6. Validar métricas do novo run em reports/metrics.json vs. baselines (Seção 2);
   comparar também com o run em Production no MLflow (experimento recsys-ecommerce).
7. Se o novo modelo for superior: promover via
   `poetry run python scripts/promote_model.py` (registra nova versão e promove a Production).
8. Se a queda persistir após re-treino: rollback (Seção 6) e abrir investigação de dados.
```

### Incidente: Cobertura de catálogo < 20% / drift de popularidade

```
1. Medir a concentração: % das impressões vindas do top-5% de itens.
2. Verificar se o último re-treino usou dados já contaminados pelo feedback loop
   (recomendações antigas dominando os positivos).
3. Mitigação de curto prazo: re-ranking com cota de diversidade/exploração
   (ex.: reservar 2 dos 10 slots para itens fora do top de popularidade).
4. Mitigação estrutural: registrar impressões (não só cliques) para permitir
   correção de viés de exposição no próximo treino.
5. Re-treinar e comparar cobertura simulada offline antes de promover.
```

### Incidente: Job de scoring falhou / taxa de erro > 5%

```
1. Checar logs do container (docker compose logs) e status do job.
2. Erros de KeyError/ID: verificar consistência entre data/processed/mappings.json
   e o modelo carregado (mesma versão de treino?).
3. Artefato corrompido: recarregar o modelo da versão Production do Registry
   (models:/recsys-embedding-mlp/Production) em vez do arquivo local.
4. Se o problema é a nova versão do modelo: rollback (Seção 6).
5. Enquanto indisponível: servir fallback por popularidade (degradação graciosa,
   nunca vitrine vazia).
```

### Incidente: Métricas offline degradadas no re-treino

```
1. NÃO promover a nova versão (o gate é o passo 6 do playbook de CTR).
2. Comparar params.yaml e dvc.lock com o último treino bom (git diff / dvc status).
3. Verificar volumetria pós-filtros: nº de interações/usuários/itens mudou muito?
4. Auditar o split temporal (vazamento) e a amostragem negativa.
5. Se necessário, reproduzir o treino bom anterior (seeds determinísticos garantem
   reprodução exata) para isolar se a causa é código, dados ou parâmetros.
```

## 5. Cadência de Re-treino

| Trigger | Condição | Ação |
|---------|----------|------|
| **Agendado** | Semanal (dados de navegação mudam rápido em e-commerce) | `dvc repro` com dados da última janela |
| **Cold-start alto** | > 15% de usuários/itens fora do vocabulário | Re-treino antecipado para atualizar mappings e embeddings |
| **Drift detectado** | PSI de scores > 0.2 ou drift de popularidade (Seção 3) | Re-treino + revisão do re-ranking |
| **Queda de negócio** | CTR ou conversão abaixo dos thresholds da Seção 2 | Playbook de CTR (inclui re-treino) |
| **Mudança de catálogo** | Entrada/saída massiva de itens (ex.: nova categoria, fim de coleção) | Avaliar impacto e re-treinar |

**Processo padrão de re-treino e promoção:**

1. Atualizar dados de entrada e executar `dvc repro` (só reexecuta estágios afetados; hiperparâmetros em `configs/params.yaml`).
2. Conferir `reports/metrics.json` contra os pisos da Seção 2 e contra o run em Production (MLflow, experimento `recsys-ecommerce`).
3. Promover: `poetry run python scripts/promote_model.py` — registra a nova versão de `recsys-embedding-mlp` e a conduz por None → Staging → Production.
4. Acompanhar CTR/cobertura por 7 dias após a troca (janela de observação).

## 6. Rollback via Model Registry

O MLflow Model Registry mantém todas as versões de `recsys-embedding-mlp` com linhagem para o run que as gerou. Rollback é uma transição de estágio, sem re-treino:

```
1. Identificar a última versão saudável (UI do MLflow ou MlflowClient —
   versões, métricas do run de origem e histórico de estágios).
2. Transicionar a versão anterior de volta para Production:
   client.transition_model_version_stage(
       name="recsys-embedding-mlp", version=<versão_anterior>,
       stage="Production", archive_existing_versions=True)
   (a versão problemática vai para Archived)
3. Reexecutar o job de scoring apontando para models:/recsys-embedding-mlp/Production
   para rematerializar as listas top-10 com o modelo restaurado.
4. Registrar o incidente (causa, versão afetada, métricas antes/depois) e
   bloquear a versão ruim de nova promoção até correção.
5. Critério de rollback: qualquer alerta Critical da Seção 2 dentro da janela
   de observação de 7 dias pós-promoção.
```

## 7. Dashboard de Monitoramento (Proposta)

| Painel | Métricas | Visualização |
|--------|----------|--------------|
| **Técnico** | Duração/latência do job, taxa de erro, frescor, recursos | Time series + indicadores |
| **Modelo** | Distribuição de scores, PSI, cobertura de catálogo, % cold-start | Histograma + gauge + time series |
| **Popularidade** | Concentração no top-5% de itens, drift vs. baseline do deploy | Time series + heatmap por categoria |
| **Negócio** | CTR, conversão atribuída, diversidade, uplift vs. controle | Time series + scorecard semanal |
| **Ciclo de vida** | Versões no Registry, métricas offline por versão, data da promoção | Tabela (fonte: MLflow) |

### Ferramentas sugeridas

- **Logs:** ELK Stack ou CloudWatch (logs estruturados do job de scoring)
- **Métricas e alertas:** Prometheus + Grafana (alertmanager para os thresholds da Seção 2)
- **Ciclo de vida do modelo:** MLflow (Tracking + Model Registry) — já em uso no projeto
- **Experimentação online:** framework de A/B com holdout de controle (vitrine de populares)
