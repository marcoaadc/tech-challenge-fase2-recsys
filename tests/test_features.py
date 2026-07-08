"""Testes de construcao de features: split temporal e amostragem negativa."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.features.build_features import (
    UniformNegativeSampler,
    positives_by_user,
    temporal_split,
)


def _sequential_interactions(n: int) -> pd.DataFrame:
    """Cria ``n`` interacoes de um unico usuario com timestamps crescentes."""
    return pd.DataFrame({"user_idx": [0] * n, "item_idx": list(range(n)), "timestamp": list(range(n))})


def test_temporal_split_respeita_ordem_cronologica() -> None:
    """Os itens mais recentes vao para teste; os mais antigos, para treino."""
    train, val, test = temporal_split(_sequential_interactions(10), test_size=0.2, val_size=0.1)
    assert len(train) == 7
    assert len(val) == 1
    assert len(test) == 2
    assert set(test["item_idx"]) == {8, 9}
    assert set(train["item_idx"]) == set(range(7))


def test_negative_sampler_respeita_ratio_e_evita_positivos() -> None:
    """O sampler gera a quantidade certa de negativos sem colidir com itens vistos."""
    positives = pd.DataFrame({"user_idx": [0, 0], "item_idx": [0, 1]})
    seen = {0: {0, 1, 2}}
    sampler = UniformNegativeSampler(negative_ratio=3, rng=np.random.default_rng(42))
    negatives = sampler.sample(positives, seen, n_items=10)
    assert len(negatives) == 3 * 2
    assert set(negatives["item_idx"]).isdisjoint({0, 1, 2})
    assert set(negatives["user_idx"]) == {0}


def test_positives_by_user_agrupa_em_conjuntos() -> None:
    """Os itens positivos de cada usuario sao agrupados em conjuntos."""
    interactions = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [5, 7, 9]})
    result = positives_by_user(interactions)
    assert result == {0: {5, 7}, 1: {9}}
