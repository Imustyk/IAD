"""Performance layer — Polars I/O, Dask, memory, lazy views, background jobs."""
from iad.performance.dask_engine import dask_available, should_use_dask
from iad.performance.fingerprints import dataframe_fingerprint, params_fingerprint
from iad.performance.jobs import BackgroundJobRunner, JobRecord, JobStatus
from iad.performance.lazy import LazyDatasetView
from iad.performance.loader import load_uploaded_file_fast, loading_metadata
from iad.performance.memory import MemoryFootprint, optimize_dtypes, prepare_for_session
from iad.performance.polars_io import polars_available, read_csv_fast

__all__ = [
    "BackgroundJobRunner",
    "JobRecord",
    "JobStatus",
    "LazyDatasetView",
    "MemoryFootprint",
    "dataframe_fingerprint",
    "dask_available",
    "load_uploaded_file_fast",
    "loading_metadata",
    "optimize_dtypes",
    "params_fingerprint",
    "polars_available",
    "prepare_for_session",
    "read_csv_fast",
    "should_use_dask",
]
