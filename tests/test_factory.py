"""Testes da ModelFactory: criacao correta por nome e erro para tipo desconhecido."""

from __future__ import annotations

import pytest

from recsys.config.settings import TrainConfig
from recsys.models.embedding_mlp import EmbeddingMLP
from recsys.models.factory import ModelFactory


def _train_config() -> TrainConfig:
    """Cria uma configuracao de treino minima para os testes."""
    return TrainConfig(
        model_type="embedding_mlp",
        embedding_dim=16,
        hidden_dims=[8],
        dropout=0.1,
        learning_rate=0.01,
        batch_size=32,
        epochs=1,
        patience=1,
        weight_decay=0.0,
    )


def test_cria_embedding_mlp() -> None:
    """O tipo ``embedding_mlp`` usa a dimensao de embedding informada."""
    model = ModelFactory.create("embedding_mlp", n_users=5, n_items=7, params=_train_config())
    assert isinstance(model, EmbeddingMLP)
    assert model.user_embedding.embedding_dim == 16


def test_cria_variante_mlp_com_embedding_menor() -> None:
    """A variante ``mlp`` reduz a dimensao dos embeddings."""
    model = ModelFactory.create("mlp", n_users=5, n_items=7, params=_train_config())
    assert isinstance(model, EmbeddingMLP)
    assert model.user_embedding.embedding_dim == 8


def test_tipo_desconhecido_levanta_erro() -> None:
    """Um tipo nao registrado deve levantar ``ValueError``."""
    with pytest.raises(ValueError, match="desconhecido"):
        ModelFactory.create("inexistente", n_users=5, n_items=7, params=_train_config())
