"""Authentification JWT — inscription, connexion, utilisateurs (JSON local)."""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator

_BACKEND_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BACKEND_DIR / "data"
_USERS_FILE = _DATA_DIR / "users.json"

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production-use-long-random-string")
JWT_DEFAULT_SECRET = "change-me-in-production-use-long-random-string"
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 h
AUTH_DISABLED = os.getenv("AUTH_DISABLED", "false").lower() in ("1", "true", "yes")

INIT_ADMIN_EMAIL = os.getenv("INIT_ADMIN_EMAIL", "admin@automl.local")
INIT_ADMIN_PASSWORD = os.getenv("INIT_ADMIN_PASSWORD", "admin123")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

auth_router = APIRouter(prefix="/auth", tags=["Authentification"])

# Format permissif : EmailStr (email-validator 2.x) rejette admin@*.local (domaine réservé).
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _validate_email_format(value: str) -> str:
    email = value.strip().lower()
    if len(email) > 254 or not _EMAIL_RE.match(email):
        raise ValueError("Adresse email invalide.")
    return email


class UserRegister(BaseModel):
    email: str
    password: str = Field(min_length=6, max_length=128)
    full_name: Optional[str] = Field(default=None, max_length=120)

    @field_validator("email")
    @classmethod
    def email_ok(cls, v: str) -> str:
        return _validate_email_format(v)


class UserLogin(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_ok(cls, v: str) -> str:
        return _validate_email_format(v)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


def _ensure_data_dir() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_users() -> List[Dict[str, Any]]:
    _ensure_data_dir()
    if not _USERS_FILE.exists():
        return []
    try:
        data = json.loads(_USERS_FILE.read_text(encoding="utf-8"))
        return list(data.get("users", []))
    except (json.JSONDecodeError, OSError):
        return []


def _save_users(users: List[Dict[str, Any]]) -> None:
    _ensure_data_dir()
    _USERS_FILE.write_text(
        json.dumps({"users": users}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _hash_password(password: str) -> str:
    return pwd_context.hash(password)


def _verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _user_public(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": user["id"],
        "email": user["email"],
        "full_name": user.get("full_name"),
        "created_at": user.get("created_at"),
    }


def create_access_token(subject: str, extra: Optional[Dict] = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": subject, "exp": expire}
    if extra:
        payload.update(extra)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def _find_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    norm = _normalize_email(email)
    for u in _load_users():
        if u.get("email") == norm:
            return u
    return None


def _find_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    for u in _load_users():
        if u.get("id") == user_id:
            return u
    return None


def init_default_admin() -> None:
    """Crée un compte admin si aucun utilisateur n'existe."""
    users = _load_users()
    if users:
        return
    if not INIT_ADMIN_PASSWORD or len(INIT_ADMIN_PASSWORD) < 6:
        return
    users.append({
        "id": str(uuid.uuid4()),
        "email": _normalize_email(INIT_ADMIN_EMAIL),
        "password_hash": _hash_password(INIT_ADMIN_PASSWORD),
        "full_name": "Administrateur",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_users(users)


def register_user(data: UserRegister) -> Dict[str, Any]:
    email = _normalize_email(data.email)
    if _find_user_by_email(email):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cet email est déjà utilisé.")
    user = {
        "id": str(uuid.uuid4()),
        "email": email,
        "password_hash": _hash_password(data.password),
        "full_name": (data.full_name or "").strip() or None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    users = _load_users()
    users.append(user)
    _save_users(users)
    return _user_public(user)


def authenticate_user(email: str, password: str) -> Dict[str, Any]:
    user = _find_user_by_email(email)
    if not user or not _verify_password(password, user["password_hash"]):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Email ou mot de passe incorrect.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Dict[str, Any]:
    if AUTH_DISABLED:
        return {"id": "dev", "email": "dev@local", "full_name": "Dev (auth off)"}

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Token manquant — connectez-vous.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token invalide.")
    except JWTError:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Token expiré ou invalide.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    user = _find_user_by_id(str(user_id))
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Utilisateur introuvable.")
    return _user_public(user)


@auth_router.post("/register", response_model=TokenResponse)
def register(data: UserRegister):
    user = register_user(data)
    token = create_access_token(user["id"], {"email": user["email"]})
    return TokenResponse(
        access_token=token,
        expires_in=JWT_EXPIRE_MINUTES * 60,
        user=user,
    )


@auth_router.post("/login", response_model=TokenResponse)
def login(data: UserLogin):
    user = authenticate_user(data.email, data.password)
    pub = _user_public(user)
    token = create_access_token(pub["id"], {"email": pub["email"]})
    return TokenResponse(
        access_token=token,
        expires_in=JWT_EXPIRE_MINUTES * 60,
        user=pub,
    )


@auth_router.get("/me")
def me(user: Dict[str, Any] = Depends(get_current_user)):
    return user


@auth_router.get("/config")
def auth_config():
    return {
        "auth_enabled": not AUTH_DISABLED,
        "token_expire_minutes": JWT_EXPIRE_MINUTES,
    }
