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
    models_dir: str
    model_path: str
    preload_first_model: bool
    model_profile: str
    server_port: int
    session_timeout: int
    max_active_conversations: int
    max_concurrent_generations: int
    max_num_images: int
    context_rollover_threshold_tokens: int
    context_rollover_recent_messages: int
    context_rollover_recent_token_budget: int
    context_rollover_headroom_tokens: int
    cpu_threads: int
    max_num_tokens: int
    enable_xnnpack_cache: bool
    use_ringbuffers_local_attention: bool
    enable_benchmark: bool
    enable_admin_llm: bool
    thinking_token_budget: int
    enable_thinking: bool
    enable_tools: bool
    force_static_max_tokens: bool
    enable_speculative_decoding: bool
    engine_ttl: int
    enable_warm_pool: bool
    filter_thinking_from_kv_cache: bool

    @property
    def model_id(self) -> str:
        if self.model_path:
            p = Path(self.model_path)
            return p.parent.name if p.name == "model.litertlm" else p.name
        return "litert-model"

    @property
    def enable_tool_calling(self) -> bool:
        return self.enable_tools

    @property
    def engine_ttl_seconds(self) -> int:
        return self.engine_ttl


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        models_dir=os.getenv("MODELS_DIR", "/models"),
        model_path=os.getenv(
            "MODEL_PATH",
            "/models/gemma-4-E2B-it.litertlm/model.litertlm",
        ),
        preload_first_model=_env_bool("PRELOAD_FIRST_MODEL", "false"),
        model_profile=os.getenv("MODEL_PROFILE", "profiles/default.yaml"),
        server_port=int(os.getenv("SERVER_PORT", "8000")),
        session_timeout=int(os.getenv("SESSION_TIMEOUT", "600")),
        max_active_conversations=int(os.getenv("MAX_ACTIVE_CONVERSATIONS", "10")),
        max_concurrent_generations=int(os.getenv("MAX_CONCURRENT_GENERATIONS", "1")),
        max_num_images=int(os.getenv("MAX_NUM_IMAGES", "0")),
        context_rollover_threshold_tokens=int(
            os.getenv("CONTEXT_ROLLOVER_THRESHOLD_TOKENS", "2600")
        ),
        context_rollover_recent_messages=int(
            os.getenv("CONTEXT_ROLLOVER_RECENT_MESSAGES", "4")
        ),
        context_rollover_recent_token_budget=int(
            os.getenv("CONTEXT_ROLLOVER_RECENT_TOKEN_BUDGET", "1024")
        ),
        context_rollover_headroom_tokens=int(
            os.getenv("CONTEXT_ROLLOVER_HEADROOM_TOKENS", "1500")
        ),
        cpu_threads=int(os.getenv("CPU_THREADS", "4")),
        max_num_tokens=int(os.getenv("MAX_NUM_TOKENS", "4096")),
        enable_xnnpack_cache=_env_bool("ENABLE_XNNPACK_CACHE", "false"),
        use_ringbuffers_local_attention=_env_bool("USE_RINGBUFFERS_LOCAL_ATTENTION", "true"),
        enable_benchmark=_env_bool("ENABLE_BENCHMARK", "false"),
        enable_admin_llm=_env_bool("ENABLE_ADMIN_LLM", "false"),
        thinking_token_budget=int(os.getenv("THINKING_TOKEN_BUDGET", "384")),
        enable_thinking=_env_bool("ENABLE_THINKING", "false"),
        enable_tools=_env_bool("ENABLE_TOOLS", os.getenv("ENABLE_TOOL_CALLING", "true")),
        force_static_max_tokens=_env_bool("FORCE_STATIC_MAX_TOKENS", "false"),
        enable_speculative_decoding=_env_bool("ENABLE_SPECULATIVE_DECODING", "false"),
        engine_ttl=int(os.getenv("ENGINE_TTL", os.getenv("ENGINE_TTL_SECONDS", "0"))),
        enable_warm_pool=_env_bool("ENABLE_WARM_POOL", "false"),
        filter_thinking_from_kv_cache=_env_bool("FILTER_THINKING_FROM_KV_CACHE", "true"),
    )
