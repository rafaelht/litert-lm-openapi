from __future__ import annotations

import base64
import hashlib
import json
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import Any


def now_ts() -> float:
    return time.time()


class ConversationIdStrategy(ABC):
    @abstractmethod
    def build(
        self,
        *,
        api_key: str,
        model: str,
        system_prompt: str,
        first_user_message: str,
    ) -> str:
        raise NotImplementedError


class DefaultConversationIdStrategy(ConversationIdStrategy):
    def build(
        self,
        *,
        api_key: str,
        model: str,
        system_prompt: str,
        first_user_message: str,
    ) -> str:
        payload = "\n".join([api_key, model, system_prompt, first_user_message])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


conversation_id_strategy: ConversationIdStrategy = DefaultConversationIdStrategy()


def make_conversation_id(api_key: str, model: str, messages: list[dict[str, Any]]) -> str:
    system_prompt = extract_system_prompt(messages)
    first_user = extract_first_user_message(messages)
    return conversation_id_strategy.build(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        first_user_message=first_user,
    )


def extract_api_key(auth_header: str | None) -> str:
    if not auth_header:
        return "anonymous"

    parts = auth_header.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip() or "anonymous"
    return auth_header.strip() or "anonymous"


def normalize_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                text_value = part.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)
        return "\n".join(chunks)

    if isinstance(content, dict):
        if content.get("type") in {"text", "input_text"} and isinstance(content.get("text"), str):
            return content["text"]

    return ""


def _translate_content_part(part: dict[str, Any]) -> dict[str, Any] | Any:
    part_type = part.get("type")

    if part_type in {"text", "input_text"}:
        text_value = part.get("text")
        if isinstance(text_value, str):
            return {"type": "text", "text": text_value}
        return part

    if part_type == "image_url":
        image_url = part.get("image_url", {})
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if not isinstance(url, str) or not url:
            return part

        if url.startswith("data:"):
            try:
                header, data = url.split(",", 1)
            except ValueError as exc:
                raise ValueError("Invalid data URL format") from exc

            if "base64" not in header:
                raise ValueError("Unsupported data URL format (only base64 is supported)")

            # Validate base64 payload early to fail-fast on broken uploads.
            base64.b64decode(data, validate=True)
            return {"type": "image", "blob": data}

        if url.startswith(("http://", "https://")):
            with urllib.request.urlopen(url, timeout=10) as response:
                raw = response.read()
            return {"type": "image", "blob": base64.b64encode(raw).decode("utf-8")}

        path = url[7:] if url.startswith("file://") else url
        return {"type": "image", "path": path}

    if part_type == "input_audio":
        input_audio = part.get("input_audio", {})
        data = input_audio.get("data") if isinstance(input_audio, dict) else None
        if isinstance(data, str) and data:
            return {"type": "audio", "blob": data}
        return part

    return part


def translate_openai_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role", "user")
    content = message.get("content")

    if not isinstance(content, list):
        return {"role": role, "content": content}

    translated_content: list[Any] = []
    for part in content:
        if isinstance(part, dict):
            translated_content.append(_translate_content_part(part))
        else:
            translated_content.append(part)

    return {
        "role": role,
        "content": translated_content,
    }


def extract_system_prompt(messages: list[dict[str, Any]]) -> str:
    system_parts: list[str] = []
    for message in messages:
        if message.get("role") in {"system", "developer"}:
            system_parts.append(normalize_text_content(message.get("content")))
    return "\n".join([part for part in system_parts if part]).strip()


def extract_first_user_message(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return normalize_text_content(message.get("content")).strip()
    return ""


def extract_incremental_message(messages: list[dict[str, Any]]) -> str:
    if not messages:
        raise ValueError("messages must not be empty")
    return normalize_text_content(messages[-1].get("content", ""))


def extract_incremental_message_payload(messages: list[dict[str, Any]]) -> str | dict[str, Any]:
    if not messages:
        raise ValueError("messages must not be empty")

    last_message = messages[-1]
    translated = translate_openai_message(last_message)
    content = translated.get("content")

    if isinstance(content, list):
        return translated

    if isinstance(content, str):
        return content

    return normalize_text_content(content)


def bootstrap_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Formatea el historial OpenAI convirtiendo prompts del sistema en contexto inyectado 
    y filtrando solo los turnos que LiteRT entiende (user/assistant).
    """
    if len(messages) <= 1:
        return []

    history = messages[:-1]
    bootstrapped: list[dict[str, Any]] = []

    for msg in history:
        role = msg.get("role")
        if role in {"system", "developer"}:
            continue

        translated = translate_openai_message(msg)
        content = translated.get("content", "")

        if role in {"user", "assistant"}:
            bootstrapped.append({"role": role, "content": content})

    return bootstrapped


def sdk_message_to_text(message: Any) -> str:
    if isinstance(message, str):
        return message

    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            return content["text"]

    if hasattr(message, "text"):
        return str(message.text)

    try:
        return json.dumps(message, ensure_ascii=False)
    except Exception:
        return ""