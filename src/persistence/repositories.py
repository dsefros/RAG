from __future__ import annotations

from datetime import datetime
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.persistence.models import AuditEvent, Group, ReindexJob, User, UserGroupMembership


class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, username: str, password_hash: str, is_admin: bool = False, is_active: bool = True) -> User:
        user = User(username=username, password_hash=password_hash, is_admin=is_admin, is_active=is_active)
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def by_username(self, username: str) -> User | None:
        return self.session.scalar(select(User).where(User.username == username))

    def list(self) -> list[User]:
        return list(self.session.scalars(select(User).order_by(User.username)))


class GroupRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, name: str) -> Group:
        group = Group(name=name)
        self.session.add(group)
        self.session.commit()
        self.session.refresh(group)
        return group

    def list(self) -> list[Group]:
        return list(self.session.scalars(select(Group).order_by(Group.name)))

    def by_name(self, name: str) -> Group | None:
        return self.session.scalar(select(Group).where(Group.name == name))


class MembershipRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, user_id: int, group_id: int) -> UserGroupMembership:
        membership = UserGroupMembership(user_id=user_id, group_id=group_id)
        self.session.add(membership)
        self.session.commit()
        self.session.refresh(membership)
        return membership

    def remove(self, user_id: int, group_id: int) -> None:
        membership = self.session.scalar(
            select(UserGroupMembership).where(
                UserGroupMembership.user_id == user_id,
                UserGroupMembership.group_id == group_id,
            )
        )
        if membership:
            self.session.delete(membership)
            self.session.commit()

    def groups_for_user(self, user_id: int) -> list[str]:
        q = (
            select(Group.name)
            .join(UserGroupMembership, Group.id == UserGroupMembership.group_id)
            .where(UserGroupMembership.user_id == user_id)
        )
        return list(self.session.scalars(q))


class ReindexJobRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, triggered_by: int, status: str = "queued") -> ReindexJob:
        job = ReindexJob(triggered_by=triggered_by, status=status, report={})
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        return job


    def mark_running(self, job: ReindexJob) -> ReindexJob:
        job.status = "running"
        self.session.commit()
        self.session.refresh(job)
        return job

    def get_running(self) -> ReindexJob | None:
        return self.session.scalar(select(ReindexJob).where(ReindexJob.status.in_(["queued", "running"])).order_by(ReindexJob.id.desc()))
    def mark_done(self, job: ReindexJob, report: dict, activated_collection_name: str | None) -> ReindexJob:
        job.status = "succeeded"
        job.finished_at = datetime.utcnow()
        job.report = report
        job.activated_collection_name = activated_collection_name
        self.session.commit()
        self.session.refresh(job)
        return job

    def mark_failed(self, job: ReindexJob, error: str, report: dict) -> ReindexJob:
        job.status = "failed"
        job.finished_at = datetime.utcnow()
        job.error_message = error
        job.report = report
        self.session.commit()
        self.session.refresh(job)
        return job

    def get(self, job_id: int) -> ReindexJob | None:
        return self.session.get(ReindexJob, job_id)


class AuditRepository:
    def __init__(self, session: Session):
        self.session = session

    def log(self, *, request_id: str, user_id: int | None, event_type: str, question_raw: str | None, answer_raw: str | None, trace: dict) -> AuditEvent:
        event = AuditEvent(
            request_id=request_id,
            user_id=user_id,
            event_type=event_type,
            question_raw=question_raw,
            answer_raw=answer_raw,
            trace=trace,
        )
        self.session.add(event)
        self.session.commit()
        self.session.refresh(event)
        return event
