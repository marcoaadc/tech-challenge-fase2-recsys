"""Utilitario de reprodutibilidade: fixa as sementes de random, numpy e torch."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fixa as sementes de ``random``, ``numpy`` e ``torch`` para reprodutibilidade.

    Args:
        seed: Valor da semente a ser aplicado em todas as fontes de aleatoriedade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
