"""Pipeline de features: split temporal e amostragem negativa.

Uso: ``python -m recsys.pipelines.build_features``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config.settings import load_params
from recsys.features.build_features import UniformNegativeSampler, build_features
from recsys.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _load_processed(processed_dir: Path) -> tuple[pd.DataFrame, int]:
    """Carrega as interacoes e o numero de itens do diretorio de processados."""
    interactions = pd.read_parquet(processed_dir / "interactions.parquet")
    with open(processed_dir / "mappings.json", encoding="utf-8") as handle:
        n_items = json.load(handle)["n_items"]
    return interactions, n_items


def _save_splits(splits: dict[str, pd.DataFrame], features_dir: Path) -> None:
    """Salva os DataFrames de treino/val/test em parquet."""
    features_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(features_dir / f"{name}.parquet", index=False)


def main() -> None:
    """Constroi e salva os conjuntos de treino, validacao e teste."""
    params = load_params()
    set_seed(params.seed)
    interactions, n_items = _load_processed(Path(params.data.processed_dir))
    sampler = UniformNegativeSampler(params.features.negative_ratio, np.random.default_rng(params.seed))
    train, val, test = build_features(
        interactions, n_items, params.features.test_size, params.features.val_size, sampler
    )
    _save_splits({"train": train, "val": val, "test": test}, Path(params.data.features_dir))
    logger.info("Treino: %d | val: %d | test: %d", len(train), len(val), len(test))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
