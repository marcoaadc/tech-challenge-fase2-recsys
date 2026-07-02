"""Baselines de recomendacao em scikit-learn com interface comum de pontuacao."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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


class LogisticRegressionRecommender(Recommender):
    """Baseline de regressao logistica sobre atributos de frequencia de usuario e item."""

    def __init__(self, n_users: int, n_items: int) -> None:
        """Inicializa o recomendador.

        Args:
            n_users: Numero total de usuarios.
            n_items: Numero total de itens.
        """
        self.n_users = n_users
        self.n_items = n_items
        self._model = LogisticRegression(max_iter=1000)
        self._user_activity = np.zeros(n_users, dtype=np.float64)
        self._item_popularity = np.zeros(n_items, dtype=np.float64)

    def _features(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Monta a matriz de atributos ``[atividade_usuario, popularidade_item]``."""
        return np.column_stack([self._user_activity[users], self._item_popularity[items]])

    def fit(self, train: pd.DataFrame) -> LogisticRegressionRecommender:
        """Ajusta a regressao logistica usando positivos e negativos de treino.

        Args:
            train: DataFrame com ``user_idx, item_idx, label``.

        Returns:
            A propria instancia treinada.
        """
        positives = train[train["label"] == 1]
        self._user_activity = _count_by_index(positives["user_idx"], self.n_users)
        self._item_popularity = _count_by_index(positives["item_idx"], self.n_items)
        features = self._features(train["user_idx"].to_numpy(), train["item_idx"].to_numpy())
        self._model.fit(features, train["label"].to_numpy())
        return self

    def score(self, user_idx: int) -> np.ndarray:
        """Pontua todos os itens para ``user_idx`` via probabilidade da classe positiva."""
        items = np.arange(self.n_items)
        users = np.full(self.n_items, user_idx)
        return self._model.predict_proba(self._features(users, items))[:, 1]
