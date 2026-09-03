from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


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
    cpu_threads: int
    max_num_tokens: int
    enable_xnnpack_cache: bool
    use_ringbuffers_local_attention: bool
    enable_benchmark: bool
    enable_admin_llm: bool
    thinking_token_budget: int
    enable_thinking: bool
    enable_tools: bool

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
        model_profile=os.getenv("MODEL_PROFILE", "profiles/default.yaml"),
        server_port=int(os.getenv("SERVER_PORT", "8000")),
        session_timeout=int(os.getenv("SESSION_TIMEOUT", "1800")),
        max_active_conversations=int(os.getenv("MAX_ACTIVE_CONVERSATIONS", "5")),
        max_num_images=int(os.getenv("MAX_NUM_IMAGES", "0")),
        context_rollover_threshold_tokens=int(
            os.getenv("CONTEXT_ROLLOVER_THRESHOLD_TOKENS", "3400")
        ),
        context_rollover_recent_messages=int(
            os.getenv("CONTEXT_ROLLOVER_RECENT_MESSAGES", "4")
        ),
        context_rollover_recent_token_budget=int(
            os.getenv("CONTEXT_ROLLOVER_RECENT_TOKEN_BUDGET", "1024")
        ),
        cpu_threads=int(os.getenv("CPU_THREADS", "4")),
        max_num_tokens=int(os.getenv("MAX_NUM_TOKENS", "4096")),
        enable_xnnpack_cache=os.getenv("ENABLE_XNNPACK_CACHE", "false").lower()
        in {"true", "1", "yes"},
        use_ringbuffers_local_attention=os.getenv(
            "USE_RINGBUFFERS_LOCAL_ATTENTION", "true"
        ).lower()
        in {"true", "1", "yes"},
        enable_benchmark=os.getenv("ENABLE_BENCHMARK", "false").lower()
        in {"true", "1", "yes"},
        enable_admin_llm=os.getenv("ENABLE_ADMIN_LLM", "false").lower()
        in {"true", "1", "yes"},
        thinking_token_budget=int(os.getenv("THINKING_TOKEN_BUDGET", "384")),
        enable_thinking=os.getenv("ENABLE_THINKING", "false").lower()
        in {"true", "1", "yes"},
        enable_tools=os.getenv("ENABLE_TOOLS", "true").lower()
        in {"true", "1", "yes"},
    )
