"""Training subsystem — service, registry, reports, reproducibility, persistence."""
from iad.ml.training.persistence import load_bundle, save_bundle
from iad.ml.training.registry import ModelRegistry, ModelSpec, Task
from iad.ml.training.reports import LeaderboardEntry, TrainingResult
from iad.ml.training.reproducibility import (
    EnvironmentFingerprint,
    ModelCard,
    SeedManager,
    capture_environment,
    fingerprint_dataframe,
)
from iad.ml.training.service import TrainingConfig, TrainingService

__all__ = [
    "TrainingService",
    "TrainingConfig",
    "TrainingResult",
    "LeaderboardEntry",
    "ModelRegistry",
    "ModelSpec",
    "Task",
    "ModelCard",
    "EnvironmentFingerprint",
    "SeedManager",
    "capture_environment",
    "fingerprint_dataframe",
    "save_bundle",
    "load_bundle",
]
