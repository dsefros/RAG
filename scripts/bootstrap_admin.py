from __future__ import annotations

import argparse

from src.auth.security import hash_password
from src.config.settings import get_settings
from src.persistence.db import build_session_factory
from src.persistence.repositories import UserRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or update first admin user")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    settings = get_settings()
    session_factory = build_session_factory(settings)
    with session_factory() as session:
        repo = UserRepository(session)
        existing = repo.by_username(args.username)
        if existing:
            existing.password_hash = hash_password(args.password)
            existing.is_admin = True
            existing.is_active = True
            session.commit()
            print(f"updated admin user: {args.username}")
            return

        repo.create(args.username, hash_password(args.password), is_admin=True, is_active=True)
        print(f"created admin user: {args.username}")


if __name__ == "__main__":
    main()
