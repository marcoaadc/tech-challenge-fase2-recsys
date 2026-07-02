"""Testes das metricas de ranking com valores conhecidos calculados a mao."""

from __future__ import annotations

import math

import numpy as np
import pytest

from recsys.evaluation.metrics import (
    auc_score,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k() -> None:
    """Precisao = acertos no top-k dividido por k."""
    assert precision_at_k([10, 20, 30, 40], {20, 40}, k=4) == pytest.approx(0.5)


def test_recall_at_k() -> None:
    """Recall = acertos no top-k dividido pelo total de relevantes."""
    assert recall_at_k([10, 20, 30], {20, 99}, k=3) == pytest.approx(0.5)


def test_recall_at_k_sem_relevantes_retorna_zero() -> None:
    """Sem itens relevantes o recall e definido como zero."""
    assert recall_at_k([10, 20], set(), k=2) == 0.0


def test_hit_rate_at_k() -> None:
    """Hit rate e 1 quando ha acerto no top-k e 0 caso contrario."""
    assert hit_rate_at_k([10, 20, 30], {30}, k=3) == 1.0
    assert hit_rate_at_k([10, 20, 30], {99}, k=3) == 0.0


def test_ndcg_at_k_acerto_no_topo() -> None:
    """Acerto na primeira posicao produz NDCG igual a 1."""
    assert ndcg_at_k([10, 20, 30], {10}, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_acerto_na_segunda_posicao() -> None:
    """Acerto unico na segunda posicao: DCG = 1/log2(3), IDCG = 1."""
    assert ndcg_at_k([10, 20, 30], {20}, k=3) == pytest.approx(1.0 / math.log2(3))


def test_auc_score() -> None:
    """AUC de um exemplo classico vale 0.75."""
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1])
    assert auc_score(scores, labels) == pytest.approx(0.75)
