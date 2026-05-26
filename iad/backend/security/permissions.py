"""Role-based access control (RBAC)."""
from __future__ import annotations

from enum import Enum

from iad.core.exceptions import PermissionError_


class Role(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"

    @classmethod
    def from_user_flags(cls, *, is_superuser: bool, role: str | None) -> Role:
        if is_superuser:
            return cls.ADMIN
        if role:
            try:
                return cls(role)
            except ValueError:
                pass
        return cls.VIEWER


class Permission(str, Enum):
    DATASET_READ = "dataset:read"
    DATASET_WRITE = "dataset:write"
    MODEL_TRAIN = "model:train"
    MODEL_PREDICT = "model:predict"
    EXPERIMENT_READ = "experiment:read"
    ADMIN_USERS = "admin:users"


_ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: frozenset({
        Permission.DATASET_READ,
        Permission.EXPERIMENT_READ,
    }),
    Role.ANALYST: frozenset({
        Permission.DATASET_READ,
        Permission.DATASET_WRITE,
        Permission.MODEL_TRAIN,
        Permission.MODEL_PREDICT,
        Permission.EXPERIMENT_READ,
    }),
    Role.ADMIN: frozenset(Permission),
}


def permissions_for_role(role: Role | str) -> frozenset[Permission]:
    if isinstance(role, str):
        role = Role(role)
    return _ROLE_PERMISSIONS.get(role, frozenset())


def has_permission(role: Role | str, permission: Permission) -> bool:
    return permission in permissions_for_role(role)


def require_permission(role: Role | str, permission: Permission) -> None:
    if not has_permission(role, permission):
        raise PermissionError_(
            f"Role {role!s} lacks permission {permission.value}",
            user_message="You do not have permission to perform this action.",
            required_permission=permission.value,
            role=str(role),
        )
