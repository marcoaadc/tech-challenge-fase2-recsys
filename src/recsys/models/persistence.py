"""Serializacao e reconstrucao de modelos neurais treinados."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from recsys.config.settings import TrainConfig
from recsys.models.factory import ModelFactory


def _architecture(params: TrainConfig) -> dict[str, Any]:
    """Extrai os metadados de arquitetura relevantes para a reconstrucao."""
    return {"embedding_dim": params.embedding_dim, "hidden_dims": params.hidden_dims, "dropout": params.dropout}


def save_checkpoint(model: nn.Module, params: TrainConfig, n_users: int, n_items: int, path: Path | str) -> None:
    """Salva o estado do modelo junto com os metadados necessarios para recria-lo.

    Args:
        model: Modelo treinado.
        params: Hiperparametros de treino/arquitetura usados.
        n_users: Numero de usuarios.
        n_items: Numero de itens.
        path: Caminho de saida (ex.: ``models/model.pt``).
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_type": params.model_type,
        "n_users": n_users,
        "n_items": n_items,
        "arch": _architecture(params),
        "train_params": params.model_dump(),
    }
    torch.save(payload, output)


def load_checkpoint(path: Path | str) -> tuple[nn.Module, dict[str, Any]]:
    """Carrega um checkpoint e reconstroi o modelo via :class:`ModelFactory`.

    Args:
        path: Caminho do arquivo salvo por :func:`save_checkpoint`.

    Returns:
        Par ``(modelo_carregado, payload)`` com o modelo em modo de avaliacao.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    params = TrainConfig(**payload["train_params"])
    model = ModelFactory.create(payload["model_type"], payload["n_users"], payload["n_items"], params)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
