"""PL2 Access controls: role-based permissions with session checks (Table 5)."""
from __future__ import annotations
from dataclasses import dataclass
from .schema import ROLES

@dataclass
class Session:
    user: str
    role: str
    active: bool = True       # False once revoked / expired
    mfa_ok: bool = True

class AccessControl:
    def __init__(self, enforce_mfa: bool = True):
        self.enforce_mfa = enforce_mfa
    def check(self, session: Session | None, permission: str):
        """Return (allowed, reason). Denies unknown roles, inactive sessions, missing privilege."""
        if session is None or not session.active:
            return False, "no_active_session"
        if self.enforce_mfa and not session.mfa_ok:
            return False, "mfa_required"
        if permission not in ROLES.get(session.role, set()):
            return False, "missing_privilege"
        return True, "ok"
