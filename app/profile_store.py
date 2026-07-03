from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings
from app.utils import normalize_text_content

logger = logging.getLogger(__name__)


_ALLOWED_GENERATION_PARAMS = {
    "temperature",
    "top_p",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "n",
}


@dataclass(frozen=True)
class ModelProfile:
    path: str
    system_prompt: str
    memory: Any
    generation_params: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return Path(self.path).name

    def memory_as_text(self) -> str:
        if isinstance(self.memory, str):
            return self.memory.strip()
        if isinstance(self.memory, list):
            return "\n".join(str(item) for item in self.memory if str(item).strip()).strip()
        if isinstance(self.memory, dict):
            return json.dumps(self.memory, ensure_ascii=False, indent=2)
        if self.memory is None:
            return ""
        return str(self.memory).strip()


class ProfileStore:
    def __init__(self, profile: ModelProfile) -> None:
        self._profile = profile

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    def _render_profile_memory_block(self) -> str:
        memory_text = self._profile.memory_as_text()
        if not memory_text:
            return ""

        return "\n".join(
            [
                "Persistent profile memory:",
                "Use this only as background context.",
                "Do not present it as the topic of the current conversation.",
                "Do not mention or summarize it unless the user asks for it or it is directly relevant to the answer.",
                "<profile_memory>",
                memory_text,
                "</profile_memory>",
            ]
        )

    def combined_bootstrap_system_prompt(self, request_messages: list[dict[str, Any]]) -> str:
        request_system = "\n".join(
            normalize_text_content(msg.get("content"))
            for msg in request_messages
            if msg.get("role") in {"system", "developer"}
        ).strip()

        parts = [
            self._profile.system_prompt.strip(),
            self._render_profile_memory_block(),
            request_system,
        ]
        return "\n\n".join(part for part in parts if part)

    def effective_generation_params(self, request: Any) -> dict[str, Any]:
        # Request values always win over profile defaults.
        effective = dict(self._profile.generation_params)
        for key in _ALLOWED_GENERATION_PARAMS:
            request_value = getattr(request, key, None)
            if request_value is not None:
                effective[key] = request_value

        return {k: v for k, v in effective.items() if v is not None}

    def as_debug_dict(self) -> dict[str, Any]:
        return {
            "name": self._profile.name,
            "path": self._profile.path,
            "system_prompt": self._profile.system_prompt,
            "memory": self._profile.memory,
            "generation_params": self._profile.generation_params,
        }


_profile_store: ProfileStore | None = None
_store_lock = asyncio.Lock()


def _resolve_profile_path(raw_path: str) -> Path:
    profile_path = Path(raw_path)
    if profile_path.is_absolute():
        return profile_path.resolve()

    candidate_bases = [
        Path.cwd(),
        Path(__file__).resolve().parents[1],
    ]

    for base in candidate_bases:
        candidate = (base / profile_path).resolve()
        if candidate.exists():
            return candidate

    # Preserve previous behavior for error reporting while still being deterministic.
    return (Path.cwd() / profile_path).resolve()


def _parse_profile(raw_data: dict[str, Any], profile_path: Path) -> ModelProfile:
    system_prompt = raw_data.get("system_prompt", "")
    if not isinstance(system_prompt, str):
        raise ValueError("profile.system_prompt must be a string")

    memory = raw_data.get("memory", "")

    generation_params = raw_data.get("generation", {})
    if generation_params is None:
        generation_params = {}
    if not isinstance(generation_params, dict):
        raise ValueError("profile.generation must be an object")

    filtered_generation_params = {
        key: value
        for key, value in generation_params.items()
        if key in _ALLOWED_GENERATION_PARAMS and value is not None
    }

    return ModelProfile(
        path=str(profile_path),
        system_prompt=system_prompt.strip(),
        memory=memory,
        generation_params=filtered_generation_params,
    )


def _load_profile() -> ProfileStore:
    settings = get_settings()
    profile_path = _resolve_profile_path(settings.model_profile)

    if not profile_path.exists():
        raise FileNotFoundError(
            "Profile file not found: "
            f"{profile_path}. "
            "Set MODEL_PROFILE to an absolute path or ensure profiles/ is copied into the image."
        )

    with profile_path.open("r", encoding="utf-8") as profile_file:
        raw_data = yaml.safe_load(profile_file) or {}

    if not isinstance(raw_data, dict):
        raise ValueError("Profile YAML root must be an object")

    profile = _parse_profile(raw_data, profile_path)
    logger.info(
        "Model profile loaded: %s (generation defaults: %s)",
        profile.path,
        sorted(profile.generation_params.keys()),
    )
    return ProfileStore(profile)


async def init_profile_store() -> ProfileStore:
    global _profile_store

    if _profile_store is not None:
        return _profile_store

    async with _store_lock:
        if _profile_store is None:
            _profile_store = _load_profile()
        return _profile_store


def get_profile_store() -> ProfileStore:
    if _profile_store is None:
        raise RuntimeError("Profile store is not initialized")
    return _profile_store
