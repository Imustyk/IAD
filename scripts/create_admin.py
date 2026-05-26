#!/usr/bin/env python3
"""Bootstrap an admin user (development / ops).

Usage:
    python scripts/create_admin.py --email admin@example.com --password 'SecurePass1'
"""
from __future__ import annotations

import argparse
import sys
from getpass import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create IAD admin user")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", default=None)
    parser.add_argument("--full-name", default="Administrator")
    args = parser.parse_args()

    password = args.password or getpass("Password: ")

    from iad.backend.database.init_db import create_all_tables
    from iad.backend.database.session import session_scope
    from iad.backend.repositories.unit_of_work import UnitOfWork
    from iad.backend.security.passwords import hash_password
    from iad.config.settings import get_settings

    settings = get_settings()
    create_all_tables(settings)

    with session_scope() as session:
        uow = UnitOfWork(session)
        existing = uow.users.get_by_email(args.email)
        if existing:
            existing.hashed_password = hash_password(password)
            existing.is_superuser = True
            existing.role = "admin"
            existing.is_active = True
            print(f"Updated admin: {existing.email} ({existing.id})")
        else:
            user = uow.users.create(
                email=args.email,
                full_name=args.full_name,
                hashed_password=hash_password(password),
                is_superuser=True,
                role="admin",
            )
            print(f"Created admin: {user.email} ({user.id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
