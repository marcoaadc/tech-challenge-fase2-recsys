"""Testes dos baselines: item-kNN pontua itens co-ocorrentes acima de isolados."""

from __future__ import annotations

import pandas as pd

from recsys.models.baselines import ItemKnnRecommender, PopularityRecommender


def _train_frame() -> pd.DataFrame:
    """Cria interacoes sinteticas onde os itens 0 e 1 co-ocorrem e o 2 fica isolado."""
    rows = [
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
        (2, 2, 1),
        (3, 0, 1),
        (3, 2, 0),  # negativo: deve ser ignorado na matriz de interacoes
    ]
    return pd.DataFrame(rows, columns=["user_idx", "item_idx", "label"])


def test_score_tem_tamanho_n_items() -> None:
    """O vetor de escores cobre todos os itens do catalogo."""
    recommender = ItemKnnRecommender(n_users=4, n_items=3).fit(_train_frame())
    scores = recommender.score(0)
    assert scores.shape == (3,)


def test_item_similar_pontua_acima_de_isolado() -> None:
    """Para quem viu o item 0, o item co-ocorrente (1) supera o isolado (2)."""
    recommender = ItemKnnRecommender(n_users=4, n_items=3).fit(_train_frame())
    scores = recommender.score(3)
    assert scores[1] > scores[2]
    assert scores[2] == 0.0


def test_usuario_sem_historico_recebe_zeros() -> None:
    """Um usuario sem positivos de treino nao acumula similaridade alguma."""
    recommender = ItemKnnRecommender(n_users=5, n_items=3).fit(_train_frame())
    scores = recommender.score(4)
    assert scores.tolist() == [0.0, 0.0, 0.0]


def test_popularity_ordena_por_frequencia() -> None:
    """A popularidade reflete a contagem de positivos por item."""
    recommender = PopularityRecommender(n_items=3).fit(_train_frame())
    scores = recommender.score(0)
    assert scores[0] > scores[2]
