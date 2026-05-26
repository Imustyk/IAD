"""Frontend service layer — bridges UI to domain / ML code."""
from iad.frontend.services.context import (
    SESSION_KEYS,
    ensure_session,
    get_dataframe,
    require_dataframe,
    store_dataset,
)
from iad.frontend.services.training import (
    UnifiedTrainingReport,
    train_enterprise,
    train_legacy,
)

__all__ = [
    "SESSION_KEYS",
    "UnifiedTrainingReport",
    "ensure_session",
    "get_dataframe",
    "require_dataframe",
    "store_dataset",
    "train_enterprise",
    "train_legacy",
]
