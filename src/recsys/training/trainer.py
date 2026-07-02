"""Loop de treino com validacao por epoca, early stopping e logging no MLflow."""

from __future__ import annotations

import logging
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

ValidateFn = Callable[[], float]


@dataclass
class TrainingConfig:
    """Configuracao do otimizador e do criterio de parada do treino."""

    learning_rate: float
    weight_decay: float
    epochs: int
    patience: int


@dataclass
class _EarlyStopping:
    """Rastreia a melhor metrica de validacao e o melhor estado do modelo."""

    patience: int
    best_metric: float = float("-inf")
    best_state: dict[str, torch.Tensor] | None = field(default=None)
    _no_improve: int = 0

    def update(self, metric: float, model: nn.Module) -> None:
        """Atualiza o melhor estado se a metrica melhorou; senao conta a estagnacao."""
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_state = deepcopy(model.state_dict())
            self._no_improve = 0
        else:
            self._no_improve += 1

    @property
    def should_stop(self) -> bool:
        """Indica se o treino deve parar por falta de melhora."""
        return self._no_improve >= self.patience


class Trainer:
    """Treina um modelo de recomendacao com BCE, Adam e early stopping por validacao."""

    def __init__(self, model: nn.Module, config: TrainingConfig, validate_fn: ValidateFn, device: str = "cpu") -> None:
        """Inicializa o treinador.

        Args:
            model: Modelo a ser treinado.
            config: Configuracao do otimizador e parada.
            validate_fn: Funcao sem argumentos que retorna a metrica de validacao (maior e melhor).
            device: Dispositivo de treino.
        """
        self.model = model.to(device)
        self.config = config
        self.validate_fn = validate_fn
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    def _step(self, users: torch.Tensor, items: torch.Tensor, labels: torch.Tensor) -> float:
        """Executa um passo de otimizacao e retorna a perda do lote."""
        users, items, labels = users.to(self.device), items.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(users, items), labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_epoch(self, loader: DataLoader) -> float:
        """Treina uma epoca completa e retorna a perda media ponderada."""
        self.model.train()
        total = 0.0
        for users, items, labels in loader:
            total += self._step(users, items, labels) * len(labels)
        return total / len(loader.dataset)

    @staticmethod
    def _log_epoch(epoch: int, train_loss: float, val_metric: float) -> None:
        """Registra a perda de treino e a metrica de validacao (log e MLflow)."""
        logger.info("epoch=%d train_loss=%.4f val_metric=%.4f", epoch, train_loss, val_metric)
        if mlflow.active_run() is not None:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_metric", val_metric, step=epoch)

    def fit(self, train_loader: DataLoader) -> dict[str, object]:
        """Treina o modelo, aplicando early stopping e restaurando o melhor estado.

        Args:
            train_loader: ``DataLoader`` com os exemplos de treino rotulados.

        Returns:
            Dicionario com a melhor metrica de validacao e o historico por epoca.
        """
        stopper = _EarlyStopping(self.config.patience)
        history: list[dict[str, float]] = []
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_metric = self.validate_fn()
            self._log_epoch(epoch, train_loss, val_metric)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_metric": val_metric})
            stopper.update(val_metric, self.model)
            if stopper.should_stop:
                break
        if stopper.best_state is not None:
            self.model.load_state_dict(stopper.best_state)
        return {"best_val_metric": stopper.best_metric, "history": history}
