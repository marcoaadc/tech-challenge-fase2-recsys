# syntax=docker/dockerfile:1
#
# Imagem multi-stage otimizada:
#   1. builder  -> instala Poetry e resolve as dependencias em um virtualenv local (.venv)
#   2. runtime  -> imagem slim final, sem Poetry/toolchain, usuario nao-root
#
# Build:  docker build -t recsys:latest .
# Run:    docker run --rm -v ./data:/app/data -v ./models:/app/models recsys:latest

# --------------------------------------------------------------------------
# Stage 1: builder — resolve dependencias com Poetry
# --------------------------------------------------------------------------
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=1.8.3

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Copia apenas os manifests primeiro para aproveitar o cache de camadas
COPY pyproject.toml poetry.lock ./

# Virtualenv dentro do projeto (/app/.venv) para copiar facilmente ao runtime.
# --only main: sem dependencias de dev; --no-root: o pacote entra via PYTHONPATH.
RUN poetry config virtualenvs.in-project true \
    && poetry install --only main --no-root --no-interaction --no-ansi

# --------------------------------------------------------------------------
# Stage 2: runtime — imagem final enxuta, usuario nao-root
# --------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home appuser

WORKDIR /app

# Virtualenv pronto do builder + codigo e configs do projeto
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY configs/ ./configs/
COPY dvc.yaml ./

# Diretorios de trabalho graváveis pelo usuario nao-root (tambem usados como volumes)
RUN mkdir -p data models reports \
    && chown -R appuser:appuser /app

USER appuser

# Comando padrao: etapa de treino. Sobrescreva para rodar outras etapas, ex.:
#   docker run recsys:latest python -m recsys.pipelines.evaluate
CMD ["python", "-m", "recsys.pipelines.train"]
