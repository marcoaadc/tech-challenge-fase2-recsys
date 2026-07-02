"""Pipeline de pre-processamento: gera interacoes implicitas reindexadas.

Uso: ``python -m recsys.pipelines.preprocess``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from recsys.config.settings import load_params
from recsys.data.preprocess import (
    POSITIVE_THRESHOLD,
    RatingThresholdStrategy,
    preprocess,
    save_processed,
)
from recsys.utils.seed import set_seed

logger = logging.getLogger(__name__)

_RAW_DATA_FILE = Path("ml-100k") / "u.data"


def main() -> None:
    """Executa o pre-processamento e salva interacoes e mapeamentos."""
    params = load_params()
    set_seed(params.seed)
    raw_path = Path(params.data.raw_dir) / _RAW_DATA_FILE
    strategy = RatingThresholdStrategy(POSITIVE_THRESHOLD)
    interactions, mappings = preprocess(
        raw_path, strategy, params.data.min_user_interactions, params.data.min_item_interactions
    )
    save_processed(interactions, mappings, params.data.processed_dir)
    logger.info("Interacoes: %d | usuarios: %d | itens: %d", len(interactions), mappings.n_users, mappings.n_items)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
