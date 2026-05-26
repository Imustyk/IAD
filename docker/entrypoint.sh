#!/bin/sh
# IAD container entrypoint — wait for DB, optional migrations, then exec CMD.
set -eu

if [ "${IAD_RUN_MIGRATIONS:-false}" = "true" ]; then
  echo "[entrypoint] running alembic upgrade head"
  attempt=0
  max_attempts="${IAD_MIGRATION_RETRIES:-30}"
  until alembic upgrade head; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "[entrypoint] alembic upgrade failed after ${max_attempts} attempts" >&2
      exit 1
    fi
    echo "[entrypoint] database not ready or migration failed — retry ${attempt}/${max_attempts} in 2s"
    sleep 2
  done
  echo "[entrypoint] migrations complete"
fi

exec "$@"
