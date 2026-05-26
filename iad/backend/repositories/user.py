"""User repository."""
from __future__ import annotations

from sqlalchemy import select

from iad.backend.models.user import User
from iad.backend.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    model = User

    def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(User.email == email.lower().strip())
        return self.session.scalar(stmt)

    def create(
        self,
        *,
        email: str,
        full_name: str | None = None,
        hashed_password: str | None = None,
        is_superuser: bool = False,
        role: str = "analyst",
    ) -> User:
        user = User(
            email=email.lower().strip(),
            full_name=full_name,
            hashed_password=hashed_password,
            is_superuser=is_superuser,
            role=role,
        )
        return self.add(user)

    def update_password(self, user_id: str, hashed_password: str) -> User:
        user = self.get_or_raise(user_id)
        user.hashed_password = hashed_password
        self.session.flush()
        return user

    def set_role(self, user_id: str, role: str) -> User:
        user = self.get_or_raise(user_id)
        user.role = role
        self.session.flush()
        return user

    def get_or_create_default(self, email: str = "default@iad.local") -> User:
        existing = self.get_by_email(email)
        if existing is not None:
            return existing
        return self.create(email=email, full_name="Default User")
