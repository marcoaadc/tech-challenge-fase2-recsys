"""Metricas de ranking para avaliacao de recomendadores.

Todas as metricas ``@k`` recebem uma lista ordenada de itens recomendados (do mais
para o menos relevante) e o conjunto de itens verdadeiramente relevantes do usuario.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def precision_at_k(ranked_items: Sequence[int], relevant: set[int], k: int) -> float:
    """Fracao dos ``k`` primeiros itens recomendados que sao relevantes.

    Args:
        ranked_items: Itens recomendados em ordem decrescente de escore.
        relevant: Conjunto de itens relevantes.
        k: Corte do ranking.

    Returns:
        Precisao no corte ``k`` (entre 0 e 1).
    """
    top_k = ranked_items[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(ranked_items: Sequence[int], relevant: set[int], k: int) -> float:
    """Fracao dos itens relevantes recuperada entre os ``k`` primeiros.

    Args:
        ranked_items: Itens recomendados em ordem decrescente de escore.
        relevant: Conjunto de itens relevantes.
        k: Corte do ranking.

    Returns:
        Recall no corte ``k`` (entre 0 e 1); 0 se nao houver relevantes.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked_items[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked_items: Sequence[int], relevant: set[int], k: int) -> float:
    """Ganho cumulativo descontado normalizado (NDCG) no corte ``k``.

    Args:
        ranked_items: Itens recomendados em ordem decrescente de escore.
        relevant: Conjunto de itens relevantes.
        k: Corte do ranking.

    Returns:
        NDCG no corte ``k`` (entre 0 e 1).
    """
    dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(ranked_items[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return float(dcg / idcg) if idcg > 0 else 0.0


def hit_rate_at_k(ranked_items: Sequence[int], relevant: set[int], k: int) -> float:
    """Indica se ao menos um item relevante aparece entre os ``k`` primeiros.

    Args:
        ranked_items: Itens recomendados em ordem decrescente de escore.
        relevant: Conjunto de itens relevantes.
        k: Corte do ranking.

    Returns:
        ``1.0`` se houver ao menos um acerto no top-``k``, caso contrario ``0.0``.
    """
    return 1.0 if any(item in relevant for item in ranked_items[:k]) else 0.0


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area sob a curva ROC entre escores e rotulos binarios.

    Args:
        scores: Escores continuos atribuidos a cada item.
        labels: Rotulos binarios (1 para relevante, 0 caso contrario).

    Returns:
        Valor da AUC (entre 0 e 1).
    """
    return float(roc_auc_score(labels, scores))
