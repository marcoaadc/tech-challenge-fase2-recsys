"""Baselines de recomendacao em scikit-learn com interface comum de pontuacao."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from recsys.models.base import Recommender


def _count_by_index(values: pd.Series, size: int) -> np.ndarray:
    """Conta ocorrencias de cada indice em ``[0, size)`` como vetor denso."""
    counts = np.zeros(size, dtype=np.float64)
    observed = values.value_counts()
    counts[observed.index.to_numpy()] = observed.to_numpy()
    return counts


class PopularityRecommender(Recommender):
    """Recomendador nao personalizado que ordena itens por popularidade global."""

    def __init__(self, n_items: int) -> None:
        """Inicializa o recomendador.

        Args:
            n_items: Numero total de itens.
        """
        self.n_items = n_items
        self._popularity = np.zeros(n_items, dtype=np.float64)

    def fit(self, train: pd.DataFrame) -> PopularityRecommender:
        """Estima a popularidade a partir dos positivos de treino.

        Args:
            train: DataFrame com ``user_idx, item_idx, label``.

        Returns:
            A propria instancia treinada.
        """
        positives = train[train["label"] == 1]
        self._popularity = _count_by_index(positives["item_idx"], self.n_items)
        return self

    def score(self, user_idx: int) -> np.ndarray:
        """Retorna o vetor de popularidade (identico para todos os usuarios)."""
        return self._popularity


class ItemKnnRecommender(Recommender):
    """Baseline de filtragem colaborativa item-based com similaridade coseno kNN."""

    def __init__(self, n_users: int, n_items: int, n_neighbors: int = 50) -> None:
        """Inicializa o recomendador.

        Args:
            n_users: Numero total de usuarios.
            n_items: Numero total de itens.
            n_neighbors: Quantidade de vizinhos mais proximos por item.
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_neighbors = n_neighbors
        self._interactions = csr_matrix((n_users, n_items), dtype=np.float64)
        self._similarity = csr_matrix((n_items, n_items), dtype=np.float64)

    def _build_interactions(self, train: pd.DataFrame) -> csr_matrix:
        """Monta a matriz esparsa usuario-item a partir dos positivos de treino."""
        positives = train[train["label"] == 1]
        rows = positives["user_idx"].to_numpy()
        cols = positives["item_idx"].to_numpy()
        data = np.ones(rows.shape[0], dtype=np.float64)
        return csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def _to_similarity(self, distances: np.ndarray, indices: np.ndarray) -> csr_matrix:
        """Converte distancias/indices dos vizinhos numa matriz esparsa de similaridade."""
        rows = np.repeat(np.arange(self.n_items), indices.shape[1])
        sims = np.nan_to_num(1.0 - distances.ravel(), nan=0.0)
        similarity = csr_matrix((sims, (rows, indices.ravel())), shape=(self.n_items, self.n_items))
        similarity.setdiag(0.0)
        similarity.eliminate_zeros()
        return similarity

    def _fit_similarity(self, interactions: csr_matrix) -> csr_matrix:
        """Calcula a similaridade coseno item-item retendo apenas os ``n_neighbors``."""
        item_vectors = interactions.T.tocsr()
        k = min(self.n_neighbors + 1, self.n_items)
        knn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(item_vectors)
        distances, indices = knn.kneighbors(item_vectors)
        return self._to_similarity(distances, indices)

    def fit(self, train: pd.DataFrame) -> ItemKnnRecommender:
        """Constroi a matriz de interacoes e a similaridade item-item.

        Args:
            train: DataFrame com ``user_idx, item_idx, label``.

        Returns:
            A propria instancia treinada.
        """
        self._interactions = self._build_interactions(train)
        self._similarity = self._fit_similarity(self._interactions)
        return self

    def score(self, user_idx: int) -> np.ndarray:
        """Pontua itens somando as similaridades com o historico do ``user_idx``."""
        history = self._interactions.getrow(user_idx)
        scores = history.dot(self._similarity)
        return np.asarray(scores.todense()).ravel()
