from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_path: str
    server_port: int
    session_timeout: int
    max_active_conversations: int
    max_num_images: int

    @property
    def model_id(self) -> str:
        return Path(self.model_path).parent.name or "litert-model"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        model_path=os.getenv(
            "MODEL_PATH",
            "/models/gemma-4-E2B-it.litertlm/model.litertlm",
        ),
        server_port=int(os.getenv("SERVER_PORT", "8000")),
        session_timeout=int(os.getenv("SESSION_TIMEOUT", "1800")),
        max_active_conversations=int(os.getenv("MAX_ACTIVE_CONVERSATIONS", "1000")),
        max_num_images=int(os.getenv("MAX_NUM_IMAGES", "4")),
    )
