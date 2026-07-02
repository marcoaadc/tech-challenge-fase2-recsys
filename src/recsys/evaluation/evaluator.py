"""Avaliacao full-ranking: escora todos os itens por usuario e agrega as metricas.

Para cada usuario do conjunto de avaliacao, todos os itens sao pontuados, os itens
vistos no treino sao excluidos e as metricas de ranking sao calculadas contra os
positivos retidos. As metricas sao entao promediadas entre os usuarios.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import nn

from recsys.evaluation.metrics import (
    auc_score,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recsys.models.base import Recommender


class TorchRecommender(Recommender):
    """Adaptador que expoe um ``nn.Module`` treinado como :class:`Recommender`."""

    def __init__(self, model: nn.Module, n_items: int, device: str = "cpu") -> None:
        """Inicializa o adaptador.

        Args:
            model: Modelo neural treinado.
            n_items: Numero total de itens.
            device: Dispositivo de inferencia.
        """
        self.model = model
        self.n_items = n_items
        self.device = device
        self._items = torch.arange(n_items, device=device)

    def score(self, user_idx: int) -> np.ndarray:
        """Pontua todos os itens para ``user_idx`` retornando os logits do modelo."""
        self.model.eval()
        users = torch.full((self.n_items,), user_idx, dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(users, self._items)
        return logits.cpu().numpy()


def _user_auc(masked_scores: np.ndarray, relevant: set[int], seen: set[int]) -> float | None:
    """Calcula a AUC de um usuario sobre os itens nao vistos; ``None`` se classe unica."""
    valid = np.ones(masked_scores.shape[0], dtype=bool)
    if seen:
        valid[list(seen)] = False
    labels = np.zeros(masked_scores.shape[0])
    labels[list(relevant)] = 1.0
    labels, scores = labels[valid], masked_scores[valid]
    if labels.min() == labels.max():
        return None
    return auc_score(scores, labels)


def _accumulate_user(
    accum: dict[str, list[float]], scores: np.ndarray, relevant: set[int], seen: set[int], k: int
) -> None:
    """Calcula e acumula as metricas de um unico usuario."""
    masked = scores.astype(float).copy()
    if seen:
        masked[list(seen)] = -np.inf
    ranked = np.argsort(-masked)[:k].tolist()
    accum["precision_at_k"].append(precision_at_k(ranked, relevant, k))
    accum["recall_at_k"].append(recall_at_k(ranked, relevant, k))
    accum["ndcg_at_k"].append(ndcg_at_k(ranked, relevant, k))
    accum["hit_rate_at_k"].append(hit_rate_at_k(ranked, relevant, k))
    user_auc = _user_auc(masked, relevant, seen)
    if user_auc is not None:
        accum["auc"].append(user_auc)


def evaluate_recommender(
    recommender: Recommender,
    eval_positives: dict[int, set[int]],
    seen: dict[int, set[int]],
    k: int,
) -> dict[str, float]:
    """Avalia um recomendador com full-ranking e retorna as metricas medias.

    Args:
        recommender: Recomendador a avaliar (implementa ``score``).
        eval_positives: Itens relevantes retidos por usuario.
        seen: Itens vistos no treino por usuario (excluidos do ranking).
        k: Corte do ranking.

    Returns:
        Dicionario ``metrica -> valor medio`` entre os usuarios avaliados.
    """
    accum: dict[str, list[float]] = defaultdict(list)
    for user, relevant in eval_positives.items():
        if not relevant:
            continue
        scores = recommender.score(user)
        _accumulate_user(accum, scores, relevant, seen.get(user, set()), k)
    return {name: float(np.mean(values)) for name, values in accum.items() if values}
