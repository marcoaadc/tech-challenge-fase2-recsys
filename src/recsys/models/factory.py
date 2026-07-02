"""Fabrica de modelos neurais (**padrao Factory**).

A :class:`ModelFactory` cria instancias de ``nn.Module`` a partir de um nome de tipo
e dos hiperparametros, desacoplando o codigo cliente das classes concretas.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from recsys.config.settings import TrainConfig
from recsys.models.embedding_mlp import EmbeddingMLP

ModelBuilder = Callable[[int, int, TrainConfig], nn.Module]

_MIN_EMBEDDING_DIM = 8


def _build_embedding_mlp(n_users: int, n_items: int, params: TrainConfig) -> nn.Module:
    """Cria o modelo completo ``EmbeddingMLP`` com a configuracao informada."""
    return EmbeddingMLP(n_users, n_items, params.embedding_dim, params.hidden_dims, params.dropout)


def _build_mlp(n_users: int, n_items: int, params: TrainConfig) -> nn.Module:
    """Cria uma variante mais simples: embeddings menores e uma unica camada oculta."""
    embedding_dim = max(_MIN_EMBEDDING_DIM, params.embedding_dim // 2)
    hidden_dims = params.hidden_dims[:1] or [embedding_dim]
    return EmbeddingMLP(n_users, n_items, embedding_dim, hidden_dims, params.dropout)


class ModelFactory:
    """Fabrica que cria modelos registrados por nome."""

    _builders: dict[str, ModelBuilder] = {
        "embedding_mlp": _build_embedding_mlp,
        "mlp": _build_mlp,
    }

    @classmethod
    def register(cls, model_type: str, builder: ModelBuilder) -> None:
        """Registra um novo construtor de modelo.

        Args:
            model_type: Nome do tipo de modelo.
            builder: Funcao que constroi o modelo.
        """
        cls._builders[model_type] = builder

    @classmethod
    def available(cls) -> list[str]:
        """Retorna os tipos de modelo registrados."""
        return sorted(cls._builders)

    @classmethod
    def create(cls, model_type: str, n_users: int, n_items: int, params: TrainConfig) -> nn.Module:
        """Cria um modelo a partir do seu nome.

        Args:
            model_type: Nome do tipo registrado (ex.: ``embedding_mlp``, ``mlp``).
            n_users: Numero de usuarios.
            n_items: Numero de itens.
            params: Hiperparametros de treino/arquitetura.

        Returns:
            Instancia de ``nn.Module``.

        Raises:
            ValueError: Se ``model_type`` nao estiver registrado.
        """
        if model_type not in cls._builders:
            raise ValueError(f"Tipo de modelo desconhecido: {model_type!r}. Disponiveis: {cls.available()}")
        return cls._builders[model_type](n_users, n_items, params)
