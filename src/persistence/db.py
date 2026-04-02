from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import Settings


def build_engine(settings: Settings):
    return create_engine(settings.postgres.dsn, pool_pre_ping=True)


def build_session_factory(settings: Settings):
    engine = build_engine(settings)
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
