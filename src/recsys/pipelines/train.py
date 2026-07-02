"""Pipeline de treino: treina o modelo neural e registra o experimento no MLflow.

Uso: ``python -m recsys.pipelines.train``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
import pandas as pd

from recsys.config.settings import Params, Settings, TrainConfig, load_params
from recsys.evaluation.evaluator import TorchRecommender, evaluate_recommender
from recsys.features.build_features import positives_by_user
from recsys.models.factory import ModelFactory
from recsys.models.persistence import save_checkpoint
from recsys.training.dataset import make_dataloader
from recsys.training.trainer import Trainer, TrainingConfig, ValidateFn
from recsys.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _load_cardinalities(processed_dir: Path) -> tuple[int, int]:
    """Le ``n_users`` e ``n_items`` do arquivo de mapeamentos."""
    with open(processed_dir / "mappings.json", encoding="utf-8") as handle:
        mappings = json.load(handle)
    return mappings["n_users"], mappings["n_items"]


def _build_validate_fn(model: object, train: pd.DataFrame, val: pd.DataFrame, n_items: int, k: int) -> ValidateFn:
    """Cria a funcao de validacao que retorna o recall@k por full-ranking."""
    recommender = TorchRecommender(model, n_items)  # type: ignore[arg-type]
    seen = positives_by_user(train[train["label"] == 1])
    val_positives = positives_by_user(val)

    def validate() -> float:
        metrics = evaluate_recommender(recommender, val_positives, seen, k)
        return metrics.get("recall_at_k", 0.0)

    return validate


def _log_params(params: Params) -> None:
    """Registra os hiperparametros de treino e a semente no MLflow."""
    mlflow.log_param("seed", params.seed)
    mlflow.log_params(params.train.model_dump())


def _training_config(train: TrainConfig) -> TrainingConfig:
    """Converte os hiperparametros em configuracao do treinador."""
    return TrainingConfig(train.learning_rate, train.weight_decay, train.epochs, train.patience)


def _setup_mlflow(settings: Settings) -> None:
    """Configura o tracking URI e o experimento do MLflow."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def _run_training(params: Params, settings: Settings) -> float:
    """Constroi os artefatos, treina o modelo e salva o checkpoint; retorna a metrica."""
    features_dir = Path(params.data.features_dir)
    train_df = pd.read_parquet(features_dir / "train.parquet")
    val_df = pd.read_parquet(features_dir / "val.parquet")
    n_users, n_items = _load_cardinalities(Path(params.data.processed_dir))
    model = ModelFactory.create(params.train.model_type, n_users, n_items, params.train)
    loader = make_dataloader(train_df, params.train.batch_size, shuffle=True)
    validate_fn = _build_validate_fn(model, train_df, val_df, n_items, params.evaluate.top_k)
    _log_params(params)
    result = Trainer(model, _training_config(params.train), validate_fn).fit(loader)
    model_path = Path(settings.models_dir) / "model.pt"
    save_checkpoint(model, params.train, n_users, n_items, model_path)
    mlflow.log_artifact(str(model_path))
    mlflow.pytorch.log_model(model, artifact_path="model")
    return float(result["best_val_metric"])


def main() -> None:
    """Treina o modelo definido em ``params.yaml`` e salva o checkpoint."""
    settings, params = Settings(), load_params()
    set_seed(params.seed)
    _setup_mlflow(settings)
    with mlflow.start_run():
        best_metric = _run_training(params, settings)
        mlflow.log_metric("best_val_recall", best_metric)
    logger.info("Melhor recall de validacao: %.4f", best_metric)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
