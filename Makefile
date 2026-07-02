# Tech Challenge FIAP Fase 02 — sistema de recomendacao
# Requer: make + git-bash (Windows) ou shell POSIX (Linux/macOS), Poetry instalado.

.PHONY: install lint format test repro mlflow-ui docker-build docker-up pipeline validate help

help:
	@echo "Alvos disponiveis:"
	@echo "  install       Instala dependencias (poetry install)"
	@echo "  lint          Checa estilo/erros com ruff"
	@echo "  format        Formata codigo e aplica fixes do ruff"
	@echo "  test          Roda a suite de testes (pytest)"
	@echo "  repro         Reproduz o pipeline DVC (dvc repro)"
	@echo "  pipeline      Roda os 5 estagios do pipeline via CLIs"
	@echo "  mlflow-ui     Sobe a UI do MLflow em http://localhost:5000"
	@echo "  docker-build  Builda a imagem Docker"
	@echo "  docker-up     Sobe mlflow + train via docker compose"
	@echo "  validate      Valida o ambiente local (scripts/validate_env.py)"

install:
	poetry install

lint:
	poetry run ruff check .

format:
	poetry run ruff format .
	poetry run ruff check --fix .

test:
	poetry run pytest

repro:
	poetry run dvc repro

mlflow-ui:
	poetry run mlflow ui

docker-build:
	docker build -t recsys:latest .

docker-up:
	docker compose up --build

pipeline:
	poetry run python -m recsys.pipelines.download
	poetry run python -m recsys.pipelines.preprocess
	poetry run python -m recsys.pipelines.build_features
	poetry run python -m recsys.pipelines.train
	poetry run python -m recsys.pipelines.evaluate

validate:
	poetry run python scripts/validate_env.py
