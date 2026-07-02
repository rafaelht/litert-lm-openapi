from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from anyio import to_thread

from app.config import get_settings
from app.conversation_manager import get_conversation_manager
from app.engine import get_engine, init_engine, update_engine_activity, check_and_consume_reload_flag
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    OpenAIModel,
    OpenAIModelListResponse,
)
from app.utils import (
    bootstrap_messages,
    extract_api_key,
    extract_incremental_message_payload,
    make_conversation_id,
    normalize_text_content,
    sdk_message_to_text,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai-compatible"])


def _sse_data(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _estimate_token_count(text: str) -> int:
    if not text:
        return 0

    try:
        engine = get_engine()
        if engine is None:
            return 0
        tokens = engine.tokenize(text)
        return len(tokens) if isinstance(tokens, list) else 0
    except Exception:
        return 0


def _generate_heuristic_title(prompt: str) -> str:
    """Genera un título dinámico, limpio y ultra-rápido basado en el texto del usuario."""
    if not prompt:
        return "Conversación General"
    
    clean_text = prompt.replace('"', '').replace("'", "").replace("`", "").strip()
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    first_line = lines[0] if lines else clean_text

    words = first_line.split()
    if not words:
        return "Conversación General"

    title_words = words[:4]
    title = " ".join(title_words)

    if len(title) > 30:
        title = title[:27] + "..."
    
    return title.strip().capitalize()


@router.get("/models", response_model=OpenAIModelListResponse)
async def list_models() -> OpenAIModelListResponse:
    settings = get_settings()
    return OpenAIModelListResponse(
        data=[
            OpenAIModel(
                id=settings.model_id,
                created=int(time.time()),
            )
        ]
    )


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
    authorization: str | None = Header(default=None),
) -> Response:
    message_dicts = [
        message.model_dump(by_alias=True, exclude_none=True)
        for message in request.messages
    ]
    if not message_dicts:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    logger.info("[DEBUG] Payload messages count: %d", len(message_dicts))

    incremental_payload = extract_incremental_message_payload(message_dicts)
    if isinstance(incremental_payload, str):
        incremental_message = incremental_payload.strip()
    else:
        incremental_message = normalize_text_content(incremental_payload.get("content", "")).strip()
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # 1. CORTOCIRCUITO: Intercepción de prompts administrativos (Títulos y Tags)
    msg_lower = incremental_message.lower()
    
    is_title_req = (
        "title" in msg_lower or 
        "creative title" in msg_lower or
        "phrase with an emoji" in msg_lower or
        (request.max_tokens is not None and request.max_tokens <= 24)
    )
    
    is_tags_req = (
        "generate 1-3 broad tags" in msg_lower or
        "tags for this conversation" in msg_lower or
        (len(message_dicts) == 1 and "tags" in msg_lower)
    )

    if is_title_req or is_tags_req:
        logger.info("[BYPASS] Interceptada petición administrativa de OpenWebUI.")
        
        if is_title_req:
            chat_title = "Conversación General"
            try:
                user_prompt = ""
                for msg in reversed(message_dicts):
                    content = normalize_text_content(msg.get("content", ""))
                    if content and not any(k in content.lower() for k in ["task:", "generate", "create a concise", "{{prompt"]):
                        user_prompt = content
                        break
                
                if not user_prompt and message_dicts:
                    user_prompt = normalize_text_content(message_dicts[0].get("content", ""))

                if "user:" in user_prompt.lower():
                    user_prompt = user_prompt.lower().split("user:")[-1].strip()

                chat_title = _generate_heuristic_title(user_prompt)
                
            except Exception as e:
                logger.error("[BYPASS ERROR] Error procesando título: %s", str(e))
                chat_title = "Conversación General"
            
            mock_payload = {"title": chat_title}

        else:
            mock_payload = ["Technology", "Code"]

        mock_json = json.dumps(mock_payload, ensure_ascii=False)
        
        if request.stream:
            async def static_stream() -> AsyncIterator[str]:
                yield _sse_data({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                })
                yield _sse_data({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": mock_json}, "finish_reason": "stop"}]
                })
                yield _sse_data("[DONE]")
            return StreamingResponse(
                static_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            )
        else:
            return JSONResponse(content={
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": mock_json},
                    "finish_reason": "stop"
                }]
            })

    # 2. FLUJO NORMAL DE CONVERSACIÓN (Aislado y Protegido)
    
    # Asegurar recarga transparente del motor si fue removido por inactividad
    engine_instance = await init_engine()

    api_key = extract_api_key(authorization)
    conversation_id = make_conversation_id(api_key, request.model, message_dicts)
    manager = get_conversation_manager()

    # Si el motor se recreó, actualizar referencias internas y limpiar el caché
    if check_and_consume_reload_flag():
        logger.info("Detectada recarga del Engine. Limpiando y reasignando referencias de C++.")
        
        if hasattr(manager, "_engine"):
            manager._engine = engine_instance
            
        if hasattr(manager, "_conversations"):
            manager._conversations.clear()
        elif hasattr(manager, "clear"):
            manager.clear()

    state = await manager.get_or_create(
        conversation_id,
        bootstrap_messages=bootstrap_messages(message_dicts),
    )
    
    update_engine_activity()

    if request.stream:

        async def event_stream() -> AsyncIterator[str]:
            async with state.lock:
                state.touch()
                update_engine_activity()

                first_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield _sse_data(first_chunk)

                try:
                    iterator = state.conversation.send_message_async(incremental_payload)
                    
                    while True:
                        disconnected = await raw_request.is_disconnected()

                        try:
                            sdk_chunk = await to_thread.run_sync(next, iterator, None)
                            if sdk_chunk is None:
                                break
                        except StopIteration:
                            break

                        state.touch()
                        update_engine_activity()
                        
                        if not disconnected:
                            text_piece = sdk_message_to_text(sdk_chunk)
                            if not text_piece:
                                continue

                            payload = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text_piece},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield _sse_data(payload)
                        
                except Exception as exc:
                    logger.exception("Streaming failed for conversation %s", conversation_id)
                    err_payload = {
                        "error": {
                            "message": str(exc),
                            "type": "internal_error",
                            "code": None,
                        }
                    }
                    yield _sse_data(err_payload)

                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield _sse_data(final_chunk)
                yield _sse_data("[DONE]")

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Bloque síncrono estándar (No-Stream)
    async with state.lock:
        state.touch()
        update_engine_activity()
        try:
            sdk_response = await asyncio.to_thread(
                state.conversation.send_message,
                incremental_payload,
            )
        except Exception as exc:
            logger.exception("Completion failed for conversation %s", conversation_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    response_text = sdk_message_to_text(sdk_response)

    prompt_text = "\n".join(normalize_text_content(msg.get("content")) for msg in message_dicts)
    prompt_tokens = _estimate_token_count(prompt_text)
    completion_tokens = _estimate_token_count(response_text)

    response = ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(content=response_text),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return JSONResponse(content=response.model_dump(by_alias=True, exclude_none=True))