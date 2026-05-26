#!/bin/sh
# IAD container entrypoint — optional DB migrations, then exec CMD.
set -eu

if [ "${IAD_RUN_MIGRATIONS:-false}" = "true" ]; then
  echo "[entrypoint] running alembic upgrade head"
  alembic upgrade head
fi

exec "$@"
