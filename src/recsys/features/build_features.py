"""Construcao de features: split temporal por usuario e amostragem negativa.

A amostragem negativa usa o **padrao Strategy** (:class:`NegativeSampler`), de forma
que a politica de geracao de negativos (uniforme, por popularidade, etc.) possa ser
trocada sem alterar o pipeline de construcao de features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["user_idx", "item_idx", "label"]


def positives_by_user(interactions: pd.DataFrame) -> dict[int, set[int]]:
    """Agrupa os itens positivos de cada usuario em conjuntos.

    Args:
        interactions: DataFrame com colunas ``user_idx`` e ``item_idx``.

    Returns:
        Dicionario ``user_idx -> conjunto de item_idx``.
    """
    grouped = interactions.groupby("user_idx")["item_idx"].apply(set)
    return {int(user): items for user, items in grouped.items()}


def _split_indices(n: int, test_size: float, val_size: float) -> tuple[int, int]:
    """Calcula os cortes ``(fim_treino, fim_val)`` para ``n`` interacoes ordenadas."""
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_test = min(n_test, n - 1)
    n_val = min(n_val, n - 1 - n_test)
    return n - n_test - n_val, n - n_test


def _split_one_user(group: pd.DataFrame, test_size: float, val_size: float) -> dict[str, pd.DataFrame]:
    """Divide as interacoes ordenadas de um usuario em treino/val/test."""
    ordered = group.sort_values("timestamp")
    train_end, val_end = _split_indices(len(ordered), test_size, val_size)
    return {
        "train": ordered.iloc[:train_end],
        "val": ordered.iloc[train_end:val_end],
        "test": ordered.iloc[val_end:],
    }


def temporal_split(
    interactions: pd.DataFrame, test_size: float, val_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide as interacoes de cada usuario cronologicamente em treino/val/test.

    Args:
        interactions: DataFrame com ``user_idx, item_idx, timestamp``.
        test_size: Fracao final de cada usuario reservada para teste.
        val_size: Fracao reservada para validacao (antes do teste).

    Returns:
        Tripla ``(treino, val, test)`` de DataFrames com ``user_idx, item_idx``.
    """
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for _, group in interactions.groupby("user_idx"):
        splits = _split_one_user(group, test_size, val_size)
        for name, frame in splits.items():
            parts[name].append(frame[["user_idx", "item_idx"]])
    return tuple(pd.concat(frames, ignore_index=True) for frames in parts.values())  # type: ignore[return-value]


class NegativeSampler(ABC):
    """Estrategia de amostragem de exemplos negativos (itens nao interagidos)."""

    @abstractmethod
    def sample(self, positives: pd.DataFrame, seen: dict[int, set[int]], n_items: int) -> pd.DataFrame:
        """Gera exemplos negativos para os positivos informados.

        Args:
            positives: DataFrame de positivos com ``user_idx, item_idx``.
            seen: Itens ja vistos por usuario (a serem evitados).
            n_items: Numero total de itens.

        Returns:
            DataFrame de negativos com ``user_idx, item_idx``.
        """


class UniformNegativeSampler(NegativeSampler):
    """Amostra negativos uniformemente entre os itens nao vistos pelo usuario."""

    def __init__(self, negative_ratio: int, rng: np.random.Generator) -> None:
        """Inicializa o sampler.

        Args:
            negative_ratio: Quantidade de negativos por positivo.
            rng: Gerador de numeros aleatorios do numpy (reprodutibilidade).
        """
        self.negative_ratio = negative_ratio
        self.rng = rng

    def _sample_for_user(self, user: int, n_positives: int, forbidden: set[int], n_items: int) -> list[int]:
        """Amostra ``negative_ratio * n_positives`` itens fora de ``forbidden``."""
        needed = self.negative_ratio * n_positives
        drawn: list[int] = []
        while len(drawn) < needed:
            candidate = int(self.rng.integers(0, n_items))
            if candidate not in forbidden:
                drawn.append(candidate)
        return drawn

    def sample(self, positives: pd.DataFrame, seen: dict[int, set[int]], n_items: int) -> pd.DataFrame:
        """Gera negativos uniformes para cada usuario presente em ``positives``."""
        rows: list[dict[str, int]] = []
        for user, group in positives.groupby("user_idx"):
            forbidden = seen.get(int(user), set())
            for item in self._sample_for_user(int(user), len(group), forbidden, n_items):
                rows.append({"user_idx": int(user), "item_idx": item})
        return pd.DataFrame(rows, columns=["user_idx", "item_idx"])


def _label(frame: pd.DataFrame, value: int) -> pd.DataFrame:
    """Adiciona a coluna ``label`` com valor constante."""
    labeled = frame[["user_idx", "item_idx"]].copy()
    labeled["label"] = value
    return labeled


def build_train_set(
    train_positives: pd.DataFrame, sampler: NegativeSampler, seen: dict[int, set[int]], n_items: int
) -> pd.DataFrame:
    """Monta o conjunto de treino com positivos (label 1) e negativos (label 0).

    Args:
        train_positives: Positivos de treino com ``user_idx, item_idx``.
        sampler: Estrategia de amostragem negativa.
        seen: Itens vistos por usuario (todos os splits) a serem evitados.
        n_items: Numero total de itens.

    Returns:
        DataFrame de treino com colunas ``user_idx, item_idx, label``.
    """
    negatives = sampler.sample(train_positives, seen, n_items)
    combined = pd.concat([_label(train_positives, 1), _label(negatives, 0)], ignore_index=True)
    return combined[FEATURE_COLUMNS]


def build_features(
    interactions: pd.DataFrame,
    n_items: int,
    test_size: float,
    val_size: float,
    sampler: NegativeSampler,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Executa split temporal e amostragem negativa (apenas no treino).

    Args:
        interactions: Interacoes positivas reindexadas.
        n_items: Numero total de itens.
        test_size: Fracao de teste por usuario.
        val_size: Fracao de validacao por usuario.
        sampler: Estrategia de amostragem negativa.

    Returns:
        Tripla ``(train, val, test)`` com colunas ``user_idx, item_idx, label``.
    """
    train_pos, val_pos, test_pos = temporal_split(interactions, test_size, val_size)
    seen = positives_by_user(interactions)
    train = build_train_set(train_pos, sampler, seen, n_items)
    return train, _label(val_pos, 1), _label(test_pos, 1)
