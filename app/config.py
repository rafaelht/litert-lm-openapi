from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_DOTENV_LOADED = False


def _load_dotenv() -> None:
    global _DOTENV_LOADED

    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_path: str
    model_profile: str
    server_port: int
    session_timeout: int
    max_active_conversations: int
    max_num_images: int
    context_rollover_threshold_tokens: int
    context_rollover_recent_messages: int
    context_rollover_recent_token_budget: int
    enable_thinking: bool
    enable_tool_calling: bool
    filter_thinking_from_kv_cache: bool

    @property
    def model_id(self) -> str:
        return Path(self.model_path).parent.name or "litert-model"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        model_path=os.getenv(
            "MODEL_PATH",
            "/models/gemma-4-E2B-it.litertlm/model.litertlm",
        ),
        model_profile=os.getenv("MODEL_PROFILE", "profiles/default.yaml"),
        server_port=int(os.getenv("SERVER_PORT", "8000")),
        session_timeout=int(os.getenv("SESSION_TIMEOUT", "1800")),
        max_active_conversations=int(os.getenv("MAX_ACTIVE_CONVERSATIONS", "1000")),
        max_num_images=int(os.getenv("MAX_NUM_IMAGES", "4")),
        context_rollover_threshold_tokens=int(
            os.getenv("CONTEXT_ROLLOVER_THRESHOLD_TOKENS", "3200")
        ),
        context_rollover_recent_messages=int(
            os.getenv("CONTEXT_ROLLOVER_RECENT_MESSAGES", "2")
        ),
        context_rollover_recent_token_budget=int(
            os.getenv("CONTEXT_ROLLOVER_RECENT_TOKEN_BUDGET", "256")
        ),
        enable_thinking=_env_bool("ENABLE_THINKING"),
        enable_tool_calling=_env_bool("ENABLE_TOOL_CALLING", "true"),
        filter_thinking_from_kv_cache=_env_bool("FILTER_THINKING_FROM_KV_CACHE", "true"),
    )
