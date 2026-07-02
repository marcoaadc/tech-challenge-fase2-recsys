"""Teste do forward do EmbeddingMLP: formato do tensor de saida."""

from __future__ import annotations

import torch

from recsys.models.embedding_mlp import EmbeddingMLP


def test_forward_retorna_logit_por_par() -> None:
    """O forward produz um logit unidimensional por par (usuario, item)."""
    model = EmbeddingMLP(n_users=10, n_items=20, embedding_dim=8, hidden_dims=[16, 8], dropout=0.0)
    users = torch.tensor([0, 1, 2, 3])
    items = torch.tensor([5, 6, 7, 8])
    output = model(users, items)
    assert output.shape == (4,)
    assert output.dtype == torch.float32
