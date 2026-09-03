from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, AsyncIterator

from fastapi.responses import JSONResponse, StreamingResponse

from app.config import get_settings
from app.engine import force_garbage_collection, init_engine
from app.schemas import ChatCompletionRequest
from app.utils import normalize_text_content, sdk_message_to_text

logger = logging.getLogger(__name__)


def generate_heuristic_title(prompt: str) -> str:
    """Genera un título dinámico, limpio e instantáneo basado en el texto del usuario."""
    if not prompt:
        return "Conversación General"

    clean_text = prompt.replace('"', "").replace("'", "").replace("`", "").strip()

    # 1. Extraer la consulta real del usuario si viene dentro de una plantilla de OpenWebUI
    user_match = re.search(
        r"(?:user|human|usuario):\s*([^\n\r]+)", clean_text, flags=re.IGNORECASE
    )
    if user_match:
        clean_text = user_match.group(1).strip()
    else:
        # Filtrar líneas con prefijos de plantilla administrativa de OpenWebUI
        valid_lines: list[str] = []
        for line in clean_text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            lower = line_str.lower()
            if lower.startswith(
                (
                    "###",
                    "task:",
                    "chat:",
                    "generate",
                    "create a",
                    "summarize",
                    "title:",
                    "user:",
                )
            ):
                continue
            valid_lines.append(line_str)
        clean_text = valid_lines[0] if valid_lines else clean_text

    # 2. Limpiar posibles prefijos de plantilla o signos de puntuación iniciales
    clean_text = re.sub(r"^[#¿¡\s\-*]+", "", clean_text)
    clean_text = re.sub(
        r"^(task:|user:|prompt:)\s*", "", clean_text, flags=re.IGNORECASE
    ).strip()

    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    first_line = lines[0] if lines else clean_text

    words = first_line.split()
    if not words:
        return "Conversación General"

    title_words = words[:5]
    title = " ".join(title_words)

    if len(title) > 35:
        title = title[:32] + "..."

    title = re.sub(r"[:;,.-]+$", "", title).strip()
    return title.capitalize() if title else "Conversación General"


def generate_heuristic_tags(prompt: str) -> list[str]:
    """Genera 1 a 3 etiquetas generales basadas en palabras clave del texto."""
    if not prompt:
        return ["General"]

    lower = prompt.lower()
    tags: list[str] = []

    categories = [
        (["python", "fastapi", "javascript", "react", "c++", "código", "programar", "bug", "sql", "api"], "Programación"),
        (["docker", "linux", "servidor", "nas", "ugreen", "red", "port", "ssh"], "Sistemas"),
        (["avión", "vuelo", "motor", "física", "ciencia", "historia"], "Ciencia"),
        (["resumen", "explicar", "definición", "concepto", "cómo"], "Consulta"),
    ]

    for keywords, tag in categories:
        if any(k in lower for k in keywords):
            tags.append(tag)
            if len(tags) >= 3:
                break

    return tags if tags else ["General"]


def extract_title_from_llm_response(raw_text: str) -> str:
    """Extrae de manera robusta el título del JSON o texto retornado por el modelo."""
    if not raw_text:
        return "Conversación General"

    try:
        data = json.loads(raw_text)
        if isinstance(data, dict) and "title" in data:
            return str(data["title"]).strip()
    except Exception:
        pass

    match = re.search(r'\"title\"\s*:\s*\"([^\"]+)\"', raw_text)
    if match:
        return match.group(1).strip()

    cleaned = re.sub(r"```(?:json)?|```", "", raw_text).strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        first = (
            lines[0]
            .replace('"', "")
            .replace("{", "")
            .replace("}", "")
            .replace("title:", "")
            .strip()
        )
        return first if first else "Conversación General"
    return "Conversación General"


def detect_admin_request(
    message_dicts: list[dict[str, Any]],
    incremental_message: str,
    max_tokens: int | None,
) -> tuple[bool, str]:
    """Determina si la petición entrante es una solicitud de metadatos de OpenWebUI (Título o Tags)."""
    msg_lower = incremental_message.lower()

    is_tags = (
        "generate 1-3 broad tags" in msg_lower
        or "tags for this conversation" in msg_lower
        or (len(message_dicts) == 1 and "tags" in msg_lower and (max_tokens is None or max_tokens <= 32))
    )
    if is_tags:
        return True, "tags"

    is_title = (
        "title" in msg_lower
        or "creative title" in msg_lower
        or "phrase with an emoji" in msg_lower
        or (max_tokens is not None and max_tokens <= 24)
    )
    if is_title:
        return True, "title"

    return False, ""


def _extract_user_context(message_dicts: list[dict[str, Any]]) -> str:
    for msg in reversed(message_dicts):
        role = msg.get("role")
        content = normalize_text_content(msg.get("content", ""))
        if role == "user" and content:
            user_match = re.search(
                r"(?:user|human|usuario):\s*([^\n\r]+)", content, flags=re.IGNORECASE
            )
            if user_match:
                return user_match.group(1).strip()
            if not any(
                k in content.lower()
                for k in ["generate a concise", "task:", "{{prompt", "### task"]
            ):
                return content
    if message_dicts:
        return normalize_text_content(message_dicts[0].get("content", ""))
    return ""


async def handle_admin_completion(
    request: ChatCompletionRequest,
    req_type: str,
    message_dicts: list[dict[str, Any]],
    incremental_message: str,
    created: int,
    completion_id: str,
) -> StreamingResponse | JSONResponse:
    """Gestiona la respuesta para peticiones de título o etiquetas de OpenWebUI."""
    settings = get_settings()
    user_context = _extract_user_context(message_dicts) or incremental_message

    if req_type == "title":
        if settings.enable_admin_llm:
            chat_title = "Conversación General"
            title_conv = None
            try:
                engine = await init_engine()
                title_conv = engine.create_conversation(
                    system_message='Resume en un título de 3 a 5 palabras en formato JSON: {"title": "..."}.',
                    max_output_tokens=25,
                )
                title_response = await asyncio.to_thread(
                    title_conv.send_message,
                    incremental_message,
                )
                raw_text = sdk_message_to_text(title_response)
                chat_title = extract_title_from_llm_response(raw_text)
            except Exception as exc:
                logger.warning("Fallo en LLM title, fallback a heurística: %s", exc)
                chat_title = generate_heuristic_title(user_context)
            finally:
                if title_conv is not None and hasattr(title_conv, "close"):
                    try:
                        title_conv.close()
                    except Exception:
                        pass
                force_garbage_collection()
        else:
            chat_title = generate_heuristic_title(user_context)

        mock_payload = {"title": chat_title}
        logger.info("[TITLE] Despachado título: %s", chat_title)

    else:  # tags
        if settings.enable_admin_llm:
            tags_conv = None
            try:
                engine = await init_engine()
                tags_conv = engine.create_conversation(
                    system_message='Genera 1 a 3 etiquetas breves en lista JSON: ["tag1", "tag2"].',
                    max_output_tokens=20,
                )
                tags_response = await asyncio.to_thread(
                    tags_conv.send_message,
                    incremental_message,
                )
                raw_text = sdk_message_to_text(tags_response)
                try:
                    mock_payload = json.loads(raw_text)
                    if not isinstance(mock_payload, list):
                        mock_payload = [str(mock_payload)]
                except Exception:
                    mock_payload = generate_heuristic_tags(user_context)
            except Exception as exc:
                logger.warning("Fallo en LLM tags, fallback a heurística: %s", exc)
                mock_payload = generate_heuristic_tags(user_context)
            finally:
                if tags_conv is not None and hasattr(tags_conv, "close"):
                    try:
                        tags_conv.close()
                    except Exception:
                        pass
                force_garbage_collection()
        else:
            mock_payload = generate_heuristic_tags(user_context)

        logger.info("[TAGS] Despachadas etiquetas: %s", mock_payload)

    mock_json = json.dumps(mock_payload, ensure_ascii=False)

    if request.stream:
        async def static_stream() -> AsyncIterator[str]:
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first_chunk, ensure_ascii=False)}\n\n"

            content_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": mock_json}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            static_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return JSONResponse(
        content={
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": mock_json},
                    "finish_reason": "stop",
                }
            ],
        }
    )
