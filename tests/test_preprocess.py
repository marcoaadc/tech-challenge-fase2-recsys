"""Testes do pre-processamento: estrategias de feedback implicito e reindexacao."""

from __future__ import annotations

import pandas as pd

from recsys.data.preprocess import (
    AllInteractionsStrategy,
    RatingThresholdStrategy,
    filter_by_min_interactions,
    reindex_ids,
)


def _sample_ratings() -> pd.DataFrame:
    """Cria um DataFrame sintetico de avaliacoes com notas variadas."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": [10, 20, 10, 30],
            "rating": [5, 2, 4, 3],
            "timestamp": [100, 101, 102, 103],
        }
    )


def test_rating_threshold_strategy_mantem_positivos() -> None:
    """A estrategia de limiar mantem apenas notas maiores ou iguais a 4."""
    result = RatingThresholdStrategy(threshold=4).to_implicit(_sample_ratings())
    assert set(result["rating"]) == {4, 5}
    assert len(result) == 2


def test_all_interactions_strategy_mantem_tudo() -> None:
    """A estrategia de todas as interacoes preserva todas as linhas."""
    result = AllInteractionsStrategy().to_implicit(_sample_ratings())
    assert len(result) == 4


def test_reindex_ids_gera_indices_contiguos() -> None:
    """Ids originais sao mapeados para inteiros contiguos comecando em zero."""
    interactions = pd.DataFrame(
        {"user_id": [5, 5, 9], "item_id": [10, 40, 40], "timestamp": [1, 2, 3]}
    )
    reindexed, mappings = reindex_ids(interactions)
    assert list(reindexed.columns) == ["user_idx", "item_idx", "timestamp"]
    assert mappings.n_users == 2
    assert mappings.n_items == 2
    assert set(reindexed["user_idx"]) == {0, 1}
    assert set(reindexed["item_idx"]) == {0, 1}


def test_filter_by_min_interactions_remove_raros() -> None:
    """Usuarios e itens abaixo do minimo sao removidos iterativamente ate estabilizar."""
    interactions = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [10, 20, 10, 20, 99],
            "rating": [5, 5, 5, 5, 5],
            "timestamp": [1, 2, 3, 4, 5],
        }
    )
    result = filter_by_min_interactions(interactions, min_user=2, min_item=2)
    assert set(result["user_id"]) == {1, 2}
    assert set(result["item_id"]) == {10, 20}
    assert len(result) == 4
