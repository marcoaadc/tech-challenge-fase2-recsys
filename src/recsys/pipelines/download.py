"""Pipeline de download: baixa e extrai o dataset MovieLens 100K.

Uso: ``python -m recsys.pipelines.download``.
"""

from __future__ import annotations

import logging

from recsys.config.settings import load_params
from recsys.data.download import download_dataset

logger = logging.getLogger(__name__)


def main() -> None:
    """Baixa o dataset a partir da URL definida em ``configs/params.yaml``."""
    params = load_params()
    data_path = download_dataset(params.data.dataset_url, params.data.raw_dir)
    logger.info("Dataset disponivel em %s", data_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
