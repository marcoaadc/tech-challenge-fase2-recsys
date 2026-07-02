"""Pipeline de avaliacao: compara o modelo neural com os baselines nas mesmas metricas.

Carrega o checkpoint, avalia o modelo neural e os dois baselines com full-ranking,
grava ``reports/metrics.json`` e registra as metricas no MLflow.

Uso: ``python -m recsys.pipelines.evaluate``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
import pandas as pd

from recsys.config.settings import Params, Settings, load_params
from recsys.evaluation.evaluator import TorchRecommender, evaluate_recommender
from recsys.features.build_features import positives_by_user
from recsys.models.base import Recommender
from recsys.models.baselines import LogisticRegressionRecommender, PopularityRecommender
from recsys.models.persistence import load_checkpoint
from recsys.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _load_cardinalities(processed_dir: Path) -> tuple[int, int]:
    """Le ``n_users`` e ``n_items`` do arquivo de mapeamentos."""
    with open(processed_dir / "mappings.json", encoding="utf-8") as handle:
        mappings = json.load(handle)
    return mappings["n_users"], mappings["n_items"]


def _build_recommenders(
    model_name: str, model: object, train: pd.DataFrame, n_users: int, n_items: int
) -> dict[str, Recommender]:
    """Instancia o recomendador neural e os dois baselines ja treinados."""
    return {
        model_name: TorchRecommender(model, n_items),  # type: ignore[arg-type]
        "popularity": PopularityRecommender(n_items).fit(train),
        "logistic_regression": LogisticRegressionRecommender(n_users, n_items).fit(train),
    }


def _evaluate_all(
    recommenders: dict[str, Recommender], test_positives: dict[int, set[int]], seen: dict[int, set[int]], k: int
) -> dict[str, dict[str, float]]:
    """Avalia cada recomendador com as mesmas metricas e retorna os resultados."""
    return {name: evaluate_recommender(rec, test_positives, seen, k) for name, rec in recommenders.items()}


def _save_and_log(results: dict[str, dict[str, float]], reports_dir: Path) -> None:
    """Grava as metricas em JSON e registra cada valor no MLflow."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    for model_name, metrics in results.items():
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{model_name}.{metric_name}", value)


def _compute_results(params: Params, settings: Settings) -> dict[str, dict[str, float]]:
    """Carrega dados e modelos e produz o dicionario de metricas por recomendador."""
    features_dir = Path(params.data.features_dir)
    train_df = pd.read_parquet(features_dir / "train.parquet")
    test_df = pd.read_parquet(features_dir / "test.parquet")
    n_users, n_items = _load_cardinalities(Path(params.data.processed_dir))
    model, payload = load_checkpoint(Path(settings.models_dir) / "model.pt")
    recommenders = _build_recommenders(payload["model_type"], model, train_df, n_users, n_items)
    seen = positives_by_user(train_df[train_df["label"] == 1])
    test_positives = positives_by_user(test_df)
    return _evaluate_all(recommenders, test_positives, seen, params.evaluate.top_k)


def main() -> None:
    """Avalia o modelo neural e os baselines e persiste o relatorio de metricas."""
    settings, params = Settings(), load_params()
    set_seed(params.seed)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name="evaluation"):
        results = _compute_results(params, settings)
        _save_and_log(results, Path(settings.reports_dir))
    logger.info("Metricas salvas em %s", Path(settings.reports_dir) / "metrics.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
