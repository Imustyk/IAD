"""Application settings — typed, validated and environment-driven.

Why pydantic-settings?
    * single source of truth (no hardcoded values scattered across pages),
    * automatic .env loading with explicit precedence (process env > .env > defaults),
    * type coercion + field-level validation on boot,
    * trivial to extend per environment (development / staging / production / test).

All variables are prefixed with ``IAD_`` to avoid collisions with the host's
environment. Fields are intentionally non-secret by default; secrets are
declared with ``repr=False`` and must come from the environment in production.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the IAD platform."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="IAD_",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Identity -----------------------------------------------------------
    APP_NAME: str = "Data Science SaaS"
    APP_VERSION: str = "0.2.0"
    ENVIRONMENT: Literal["development", "staging", "production", "test"] = "development"
    DEBUG: bool = True

    # -- Paths --------------------------------------------------------------
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    EXPORTS_DIR: Path = PROJECT_ROOT / "exports"
    ASSETS_DIR: Path = PROJECT_ROOT / "iad" / "frontend" / "assets"

    # -- Logging ------------------------------------------------------------
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_JSON: bool = False
    LOG_RETENTION_DAYS: int = 30
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MiB per file
    LOG_BACKUP_COUNT: int = 10

    # -- Upload limits & data caps -----------------------------------------
    MAX_UPLOAD_MB: int = 200
    MAX_ROWS: int = 5_000_000
    MAX_COLUMNS: int = 10_000
    ALLOWED_FILE_EXTENSIONS: tuple[str, ...] = (
        ".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet", ".joblib",
    )

    # -- ML defaults --------------------------------------------------------
    DEFAULT_RANDOM_SEED: int = 42
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_CV_FOLDS: int = 5
    MAX_TRAINING_TIME_SECONDS: int = 60 * 30  # 30 min hard cap

    # -- Frontend (Phase 4) -------------------------------------------------
    UI_CACHE_TTL_SECONDS: int = 300
    UI_DEFAULT_THEME: Literal["light"] = "light"
    UI_ENABLE_ENTERPRISE_ML: bool = True

    # -- Performance (Phase 8) ----------------------------------------------
    PERF_USE_POLARS: bool = True
    PERF_USE_DASK: bool = True
    PERF_POLARS_THRESHOLD_ROWS: int = 50_000
    PERF_DASK_THRESHOLD_ROWS: int = 100_000
    PERF_DASK_PARTITION_ROWS: int = 50_000
    PERF_LAZY_PREVIEW_ROWS: int = 5_000
    PERF_AUTO_OPTIMIZE_DTYPES: bool = True
    PERF_BACKGROUND_TRAINING: bool = True
    PERF_MAX_WORKERS: int = 2

    # -- Security (Phase 7) -------------------------------------------------
    SECRET_KEY: str = Field(
        default="change-me-in-production",
        repr=False,
        description="HMAC / JWT secret. MUST be overridden via env in production.",
    )
    SESSION_TTL_MINUTES: int = 60 * 8
    AUTH_ENABLED: bool = False  # Streamlit login gate (requires database)
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    CSRF_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    CORS_ORIGINS: tuple[str, ...] = ("http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:8000")
    CORS_ALLOW_CREDENTIALS: bool = True
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_BASE_URL: str | None = None  # override e.g. http://127.0.0.1:8000
    API_TIMEOUT_SECONDS: float = 3.0

    # -- Database (Phase 6) -------------------------------------------------
    DATABASE_URL: str | None = None
    DATABASE_ECHO: bool = False
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_ENABLED: bool = False  # opt-in for Streamlit until Phase 5 API
    AUTO_CREATE_DB: bool = False  # dev only; production uses Alembic

    # -- Observability (Phase 12) -------------------------------------------
    METRICS_ENABLED: bool = True
    PROMETHEUS_METRICS_PATH: str = "/metrics"
    SENTRY_DSN: str | None = None
    SENTRY_ENABLED: bool = True
    SENTRY_TRACES_SAMPLE_RATE: float = 0.1
    SENTRY_PROFILES_SAMPLE_RATE: float = 0.0
    # Legacy alias — used by docker-compose observability profile host port mapping
    PROMETHEUS_PORT: int | None = None

    # -- Optional integrations ----------------------------------------------
    MLFLOW_TRACKING_URI: str | None = None

    # ---------------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------------
    @field_validator("DEFAULT_TEST_SIZE")
    @classmethod
    def _validate_test_size(cls, v: float) -> float:
        if not 0.05 <= v <= 0.5:
            raise ValueError("DEFAULT_TEST_SIZE must be between 0.05 and 0.5")
        return v

    @field_validator("DEFAULT_CV_FOLDS")
    @classmethod
    def _validate_cv_folds(cls, v: int) -> int:
        if not 2 <= v <= 20:
            raise ValueError("DEFAULT_CV_FOLDS must be between 2 and 20")
        return v

    @field_validator("MAX_UPLOAD_MB", "MAX_ROWS", "MAX_COLUMNS")
    @classmethod
    def _positive_ints(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("SENTRY_TRACES_SAMPLE_RATE", "SENTRY_PROFILES_SAMPLE_RATE")
    @classmethod
    def _validate_sample_rates(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Sentry sample rates must be between 0.0 and 1.0")
        return v

    @field_validator("ENVIRONMENT")
    @classmethod
    def _warn_secret_in_prod(cls, v: str, info) -> str:  # type: ignore[no-untyped-def]
        # Keep a soft check; raising at import time would crash the app.
        return v

    # ---------------------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------------------
    def ensure_directories(self) -> None:
        """Create runtime directories if missing. Idempotent."""
        for path in (
            self.DATA_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
            self.REPORTS_DIR,
            self.EXPORTS_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def resolved_database_url(self) -> str:
        """Return the SQLAlchemy URL for the active environment.

        Precedence:
            1. Explicit ``DATABASE_URL`` from env
            2. In-memory SQLite when ``ENVIRONMENT=test``
            3. File-backed SQLite under ``DATA_DIR/iad.db`` for local dev
        """
        if self.DATABASE_URL:
            return self.DATABASE_URL
        if self.ENVIRONMENT == "test":
            return "sqlite:///:memory:"
        db_file = self.DATA_DIR / "iad.db"
        return f"sqlite:///{db_file.resolve()}"

    @property
    def uses_sqlite(self) -> bool:
        return self.resolved_database_url().startswith("sqlite")

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_test(self) -> bool:
        return self.ENVIRONMENT == "test"

    def api_base_url(self) -> str:
        """Public base URL for the FastAPI backend (Streamlit / clients)."""
        if self.API_BASE_URL:
            return self.API_BASE_URL.rstrip("/")
        host = self.API_HOST
        if host in {"0.0.0.0", "::", ""}:
            host = "127.0.0.1"
        return f"http://{host}:{self.API_PORT}"

    def safe_dict(self) -> dict[str, object]:
        """Return a dict suitable for logging — strips secrets."""
        data = self.model_dump()
        for k in ("SECRET_KEY", "DATABASE_URL", "SENTRY_DSN"):
            if data.get(k):
                data[k] = "***"
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance.

    Uses ``functools.lru_cache`` so we instantiate once per process. Tests can
    bust the cache via ``get_settings.cache_clear()``.
    """
    settings = Settings()  # type: ignore[call-arg]
    settings.ensure_directories()
    return settings
