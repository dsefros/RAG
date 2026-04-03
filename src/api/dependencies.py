from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session, sessionmaker

from src.application.runtime import RuntimeContainer
from src.auth.security import decode_access_token
from src.persistence.repositories import MembershipRepository, UserRepository

security = HTTPBearer(auto_error=True)


def get_runtime_container(request: Request) -> RuntimeContainer:
    return request.app.state.container


def get_session_factory(request: Request) -> sessionmaker:
    return request.app.state.session_factory


def get_db(session_factory: sessionmaker = Depends(get_session_factory)):
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
    container: RuntimeContainer = Depends(get_runtime_container),
    db: Session = Depends(get_db),
):
    try:
        payload = decode_access_token(container.settings, creds.credentials)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    user = UserRepository(db).by_username(payload["username"])
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Inactive or missing user")
    return user


def get_effective_groups(user=Depends(get_current_user), db: Session = Depends(get_db)) -> list[str]:
    return MembershipRepository(db).groups_for_user(user.id)


def require_admin(user=Depends(get_current_user)):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin required")
    return user
