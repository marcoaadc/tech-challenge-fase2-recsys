"""Rede neural de recomendacao: embeddings de usuario e item concatenados em um MLP."""

from __future__ import annotations

import torch
from torch import nn


class EmbeddingMLP(nn.Module):
    """Modelo de recomendacao com embeddings de usuario/item seguidos de um MLP.

    Os embeddings de usuario e item sao concatenados e passados por uma pilha de
    camadas totalmente conectadas que produz um unico logit por par (usuario, item).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        """Inicializa o modelo.

        Args:
            n_users: Numero de usuarios.
            n_items: Numero de itens.
            embedding_dim: Dimensao dos embeddings de usuario e item.
            hidden_dims: Dimensoes das camadas ocultas do MLP.
            dropout: Probabilidade de dropout entre as camadas.
        """
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.mlp = self._build_mlp(2 * embedding_dim, hidden_dims, dropout)
        self._init_embeddings()

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
        """Constroi a pilha de camadas do MLP terminando em um unico logit."""
        layers: list[nn.Module] = []
        current = input_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(dropout)]
            current = hidden
        layers.append(nn.Linear(current, 1))
        return nn.Sequential(*layers)

    def _init_embeddings(self) -> None:
        """Inicializa os embeddings com uma distribuicao normal de baixa variancia."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Calcula o logit para cada par (usuario, item).

        Args:
            user_idx: Tensor de indices de usuario, shape ``(batch,)``.
            item_idx: Tensor de indices de item, shape ``(batch,)``.

        Returns:
            Tensor de logits, shape ``(batch,)``.
        """
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        features = torch.cat([user_vec, item_vec], dim=-1)
        return self.mlp(features).squeeze(-1)
