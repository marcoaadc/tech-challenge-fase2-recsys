"""Interface comum de pontuacao para avaliacao uniforme de recomendadores."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Recommender(Protocol):
    """Contrato de pontuacao usado pela avaliacao full-ranking.

    Qualquer recomendador (rede neural ou baseline) que implemente ``score`` pode ser
    avaliado com as mesmas metricas de ranking.
    """

    def score(self, user_idx: int) -> np.ndarray:
        """Pontua todos os itens para um dado usuario.

        Args:
            user_idx: Indice do usuario.

        Returns:
            Vetor de escores de tamanho ``n_items`` (maior e melhor).
        """
        ...
