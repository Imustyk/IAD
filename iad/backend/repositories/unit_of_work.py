"""Unit of Work — groups repositories in one transaction."""
from __future__ import annotations

from sqlalchemy.orm import Session

from iad.backend.repositories.dataset import DatasetRepository
from iad.backend.repositories.experiment import ExperimentRepository
from iad.backend.repositories.metric import MetricRepository
from iad.backend.repositories.ml_model import MLModelRepository
from iad.backend.repositories.prediction import PredictionRepository
from iad.backend.repositories.user import UserRepository


class UnitOfWork:
    """Expose all repositories bound to a single SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.users = UserRepository(session)
        self.datasets = DatasetRepository(session)
        self.experiments = ExperimentRepository(session)
        self.models = MLModelRepository(session)
        self.metrics = MetricRepository(session)
        self.predictions = PredictionRepository(session)

    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()

    def flush(self) -> None:
        self.session.flush()
