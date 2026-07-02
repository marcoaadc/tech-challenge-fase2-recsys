"""Configuracao de ambiente (Pydantic Settings) e loader dos hiperparametros do params.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PARAMS_PATH = Path("configs/params.yaml")


class Settings(BaseSettings):
    """Configuracoes de ambiente lidas de variaveis de ambiente ou de um arquivo ``.env``.

    Attributes:
        mlflow_tracking_uri: URI de tracking do MLflow.
        mlflow_experiment_name: Nome do experimento MLflow.
        seed: Semente global de reprodutibilidade.
        data_dir: Diretorio raiz de dados.
        models_dir: Diretorio onde os modelos sao salvos.
        reports_dir: Diretorio onde os relatorios sao salvos.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "recsys-ecommerce"
    seed: int = 42
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")


class DataConfig(BaseModel):
    """Parametros de ingestao e pre-processamento dos dados."""

    dataset_url: str
    raw_dir: Path
    processed_dir: Path
    features_dir: Path
    min_user_interactions: int
    min_item_interactions: int


class FeaturesConfig(BaseModel):
    """Parametros de construcao de features (split temporal e amostragem negativa)."""

    test_size: float
    val_size: float
    negative_ratio: int


class TrainConfig(BaseModel):
    """Hiperparametros de arquitetura e otimizacao do modelo neural."""

    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    embedding_dim: int
    hidden_dims: list[int]
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int
    weight_decay: float


class EvaluateConfig(BaseModel):
    """Parametros de avaliacao (corte de ranking)."""

    top_k: int


class Params(BaseModel):
    """Contrato completo do arquivo ``configs/params.yaml``."""

    seed: int
    data: DataConfig
    features: FeaturesConfig
    train: TrainConfig
    evaluate: EvaluateConfig


def load_params(path: Path | str = DEFAULT_PARAMS_PATH) -> Params:
    """Carrega e valida os hiperparametros a partir de um arquivo YAML.

    Args:
        path: Caminho do arquivo de parametros.

    Returns:
        Instancia validada de :class:`Params`.
    """
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Params(**raw)
