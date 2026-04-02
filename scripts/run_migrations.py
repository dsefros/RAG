from pathlib import Path

from sqlalchemy import text

from src.config.settings import get_settings
from src.persistence.db import build_engine


def main() -> None:
    settings = get_settings()
    sql = Path("migrations/001_init.sql").read_text()
    engine = build_engine(settings)
    with engine.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
    print("migrations applied")


if __name__ == "__main__":
    main()
