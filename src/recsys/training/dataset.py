"""Conversao de DataFrames de interacoes em ``DataLoader`` do PyTorch."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_dataloader(interactions: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    """Cria um ``DataLoader`` a partir de um DataFrame rotulado de interacoes.

    Args:
        interactions: DataFrame com colunas ``user_idx, item_idx, label``.
        batch_size: Tamanho do lote.
        shuffle: Se deve embaralhar os exemplos a cada epoca.

    Returns:
        ``DataLoader`` que produz tuplas ``(usuarios, itens, rotulos)``.
    """
    users = torch.tensor(interactions["user_idx"].to_numpy(), dtype=torch.long)
    items = torch.tensor(interactions["item_idx"].to_numpy(), dtype=torch.long)
    labels = torch.tensor(interactions["label"].to_numpy(), dtype=torch.float32)
    dataset = TensorDataset(users, items, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
