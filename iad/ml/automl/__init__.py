"""AutoML backends behind a clean ``AutoMLBackend`` ABC."""
from iad.ml.automl.base import AutoMLBackend, AutoMLResult
from iad.ml.automl.flaml_adapter import FLAMLBackend, flaml_available
from iad.ml.automl.pycaret_adapter import PyCaretBackend, pycaret_available

__all__ = [
    "AutoMLBackend",
    "AutoMLResult",
    "FLAMLBackend",
    "flaml_available",
    "PyCaretBackend",
    "pycaret_available",
]
