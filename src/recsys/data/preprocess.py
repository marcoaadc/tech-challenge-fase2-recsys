"""Pre-processamento: conversao para feedback implicito, filtragem e reindexacao.

O passo de conversao para feedback implicito usa o **padrao Strategy**
(:class:`ImplicitFeedbackStrategy`), permitindo trocar a regra que define o que
e uma interacao positiva sem alterar o restante do pipeline.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
POSITIVE_THRESHOLD = 4


class ImplicitFeedbackStrategy(ABC):
    """Estrategia de conversao de avaliacoes explicitas em feedback implicito."""

    @abstractmethod
    def to_implicit(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Retorna apenas as interacoes consideradas positivas.

        Args:
            interactions: DataFrame com a coluna ``rating``.

        Returns:
            Subconjunto de ``interactions`` tratado como positivo.
        """


class RatingThresholdStrategy(ImplicitFeedbackStrategy):
    """Considera positiva toda interacao com ``rating`` maior ou igual a um limiar."""

    def __init__(self, threshold: int = POSITIVE_THRESHOLD) -> None:
        """Inicializa a estrategia.

        Args:
            threshold: Nota minima para uma interacao ser positiva.
        """
        self.threshold = threshold

    def to_implicit(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Filtra interacoes com ``rating`` maior ou igual ao limiar."""
        return interactions[interactions["rating"] >= self.threshold].copy()


class AllInteractionsStrategy(ImplicitFeedbackStrategy):
    """Considera toda interacao registrada como positiva (ignora a nota)."""

    def to_implicit(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Retorna todas as interacoes como positivas."""
        return interactions.copy()


@dataclass
class Mappings:
    """Mapeamentos de ids originais para indices contiguos."""

    n_users: int
    n_items: int
    user_map: dict[int, int]
    item_map: dict[int, int]


def load_raw_interactions(path: Path | str) -> pd.DataFrame:
    """Le o arquivo ``u.data`` (separado por tabulacao) do MovieLens.

    Args:
        path: Caminho do arquivo ``u.data``.

    Returns:
        DataFrame com as colunas ``user_id, item_id, rating, timestamp``.
    """
    return pd.read_csv(path, sep="\t", names=RAW_COLUMNS)


def filter_by_min_interactions(interactions: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Filtra iterativamente usuarios e itens raros ate estabilizar.

    Args:
        interactions: DataFrame de interacoes positivas.
        min_user: Minimo de interacoes por usuario.
        min_item: Minimo de interacoes por item.

    Returns:
        DataFrame filtrado onde todo usuario e item atendem aos minimos.
    """
    current = interactions
    while True:
        valid_users = current["user_id"].value_counts().loc[lambda c: c >= min_user].index
        valid_items = current["item_id"].value_counts().loc[lambda c: c >= min_item].index
        filtered = current[current["user_id"].isin(valid_users) & current["item_id"].isin(valid_items)]
        if len(filtered) == len(current):
            return filtered.copy()
        current = filtered


def _build_index_map(values: pd.Series) -> dict[int, int]:
    """Constroi um mapa ``id_original -> indice_contiguo`` ordenado."""
    return {int(value): idx for idx, value in enumerate(sorted(values.unique()))}


def reindex_ids(interactions: pd.DataFrame) -> tuple[pd.DataFrame, Mappings]:
    """Reindexa ids de usuarios e itens para inteiros contiguos comecando em zero.

    Args:
        interactions: DataFrame filtrado de interacoes positivas.

    Returns:
        Par ``(dataframe_reindexado, mapeamentos)`` com as colunas
        ``user_idx, item_idx, timestamp``.
    """
    user_map = _build_index_map(interactions["user_id"])
    item_map = _build_index_map(interactions["item_id"])
    reindexed = pd.DataFrame(
        {
            "user_idx": interactions["user_id"].map(user_map).astype("int64"),
            "item_idx": interactions["item_id"].map(item_map).astype("int64"),
            "timestamp": interactions["timestamp"].astype("int64"),
        }
    )
    mappings = Mappings(len(user_map), len(item_map), user_map, item_map)
    return reindexed, mappings


def preprocess(
    raw_path: Path | str,
    strategy: ImplicitFeedbackStrategy,
    min_user: int,
    min_item: int,
) -> tuple[pd.DataFrame, Mappings]:
    """Executa o pre-processamento completo do dataset bruto.

    Args:
        raw_path: Caminho do arquivo ``u.data``.
        strategy: Estrategia de conversao para feedback implicito.
        min_user: Minimo de interacoes por usuario.
        min_item: Minimo de interacoes por item.

    Returns:
        Par ``(interacoes_reindexadas, mapeamentos)``.
    """
    raw = load_raw_interactions(raw_path)
    positives = strategy.to_implicit(raw)
    filtered = filter_by_min_interactions(positives, min_user, min_item)
    return reindex_ids(filtered)


def save_processed(interactions: pd.DataFrame, mappings: Mappings, output_dir: Path | str) -> None:
    """Salva as interacoes em parquet e os mapeamentos em JSON.

    Args:
        interactions: DataFrame com ``user_idx, item_idx, timestamp``.
        mappings: Mapeamentos de ids e cardinalidades.
        output_dir: Diretorio de saida (``data/processed``).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(out / "interactions.parquet", index=False)
    payload = {
        "n_users": mappings.n_users,
        "n_items": mappings.n_items,
        "user_map": mappings.user_map,
        "item_map": mappings.item_map,
    }
    with open(out / "mappings.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
