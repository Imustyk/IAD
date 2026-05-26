"""FastAPI application factory."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from iad.backend.api.routes import auth, exports, health, metrics, ml
from iad.backend.middleware import (
    CSRFMiddleware,
    PrometheusMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from iad.config.settings import get_settings
from iad.core.exceptions import (
    IADError,
)
from iad.core.logging import configure_logging, get_logger
from iad.core.observability import init_observability

logger = get_logger("iad.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging()
    init_observability(service="iad-api")
    if settings.AUTO_CREATE_DB:
        from iad.backend.database.init_db import create_all_tables

        create_all_tables(settings)
    logger.info("API started", extra=settings.safe_dict())
    yield
    logger.info("API shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=f"{settings.APP_NAME} API",
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.DEBUG else "/docs",
        redoc_url="/redoc" if settings.DEBUG else "/redoc",
        lifespan=lifespan,
    )

    # Middleware order: last added = outermost
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(CSRFMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.CORS_ORIGINS),
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(auth.router)
    app.include_router(exports.router)
    app.include_router(ml.router)

    @app.exception_handler(IADError)
    async def iad_error_handler(_request: Request, exc: IADError) -> JSONResponse:
        logger.warning("API error: %s", exc.message, extra=exc.context)
        return JSONResponse(
            status_code=exc.http_status,
            content={"code": exc.code, "message": exc.user_message, "detail": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"code": "validation_error", "errors": exc.errors()})

    return app


app = create_app()
