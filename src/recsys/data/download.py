"""Download e extracao do dataset MovieLens 100K."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 8192
_DATA_FILE = Path("ml-100k") / "u.data"


def _download_file(url: str, destination: Path) -> None:
    """Baixa um arquivo via HTTP em streaming para ``destination``."""
    logger.info("Baixando %s", url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(destination, "wb") as handle:
        for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
            handle.write(chunk)


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extrai todo o conteudo de um zip em ``target_dir``."""
    logger.info("Extraindo %s", zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)


def download_dataset(url: str, raw_dir: Path | str) -> Path:
    """Baixa e extrai o zip do MovieLens 100K em ``raw_dir``.

    Args:
        url: URL do arquivo zip do dataset.
        raw_dir: Diretorio de destino dos dados brutos.

    Returns:
        Caminho do arquivo ``u.data`` extraido.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    zip_path = raw_path / "ml-100k.zip"
    _download_file(url, zip_path)
    _extract_zip(zip_path, raw_path)
    return raw_path / _DATA_FILE
