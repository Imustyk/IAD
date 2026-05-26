#!/usr/bin/env python3
"""Initialise database schema.

Usage:
    python scripts/db_init.py              # Alembic upgrade (preferred)
    python scripts/db_init.py --create-all # metadata.create_all (dev only)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="IAD database initialisation")
    parser.add_argument(
        "--create-all",
        action="store_true",
        help="Use SQLAlchemy create_all instead of Alembic (development only)",
    )
    args = parser.parse_args()

    if args.create_all:
        from iad.backend.database.init_db import create_all_tables
        from iad.config.settings import get_settings

        create_all_tables(get_settings())
        print("Tables created via metadata.create_all")
        return 0

    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        cwd=ROOT,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
