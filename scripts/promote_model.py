"""Registra o melhor modelo do experimento no MLflow Model Registry e o promove a Production.

O fluxo segue o ciclo de vida do Registry: None -> Staging -> Production.
Uso: ``poetry run python scripts/promote_model.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.config.settings import Settings  # noqa: E402

MODEL_NAME = "recsys-embedding-mlp"
METRIC = "best_val_recall"

logger = logging.getLogger(__name__)


def find_best_run(client: MlflowClient, experiment_name: str) -> Run:
    """Retorna o run com maior valor da metrica de validacao no experimento.

    Args:
        client: Cliente do MLflow tracking.
        experiment_name: Nome do experimento a pesquisar.

    Returns:
        O run com o maior ``best_val_recall``.

    Raises:
        RuntimeError: Se o experimento nao existir ou nao tiver runs com a metrica.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experimento '{experiment_name}' nao encontrado.")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"metrics.{METRIC} > 0",
        order_by=[f"metrics.{METRIC} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"Nenhum run com a metrica '{METRIC}' encontrado.")
    return runs[0]


def register_model(run: Run) -> int:
    """Registra o artefato de modelo do run no Model Registry.

    Args:
        run: Run do MLflow contendo o artefato ``model``.

    Returns:
        A versao criada no Registry.
    """
    model_uri = f"runs:/{run.info.run_id}/model"
    version = mlflow.register_model(model_uri, MODEL_NAME)
    return int(version.version)


def promote(client: MlflowClient, version: int) -> None:
    """Promove a versao registrada seguindo o fluxo Staging -> Production.

    Args:
        client: Cliente do MLflow tracking.
        version: Versao do modelo no Registry.
    """
    for stage in ("Staging", "Production"):
        client.transition_model_version_stage(MODEL_NAME, version, stage)
        logger.info("Modelo %s v%s promovido para %s.", MODEL_NAME, version, stage)


def main() -> None:
    """Encontra o melhor run, registra o modelo e o promove a Production."""
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    best_run = find_best_run(client, settings.mlflow_experiment_name)
    metric_value = best_run.data.metrics[METRIC]
    logger.info("Melhor run: %s (%s=%.4f)", best_run.info.run_id, METRIC, metric_value)
    version = register_model(best_run)
    promote(client, version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
