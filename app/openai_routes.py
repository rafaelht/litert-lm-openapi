from __future__ import annotations

import asyncio
import inspect
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
from app.profile_store import get_profile_store
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    OpenAIModel,
    OpenAIModelListResponse,
)
from app.tooling import normalize_openai_tools, tools_signature
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


def _build_method_kwargs(method: Any, generation_params: dict[str, Any]) -> dict[str, Any]:
    if not generation_params:
        return {}

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return {}

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return generation_params

    return {
        key: value
        for key, value in generation_params.items()
        if key in signature.parameters
    }


def _sse_data(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _json_signature(value: Any) -> str:
    if not value:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _build_extra_context(settings: Any, request: ChatCompletionRequest) -> dict[str, Any]:
    extra_context: dict[str, Any] = {}
    if settings.enable_thinking:
        extra_context["enable_thinking"] = True
    if request.tool_choice is not None:
        extra_context["tool_choice"] = request.tool_choice
    return extra_context


def _normalize_tool_calls(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []

    tool_calls = message.get("tool_calls")
    if not tool_calls:
        content = message.get("content")
        if isinstance(content, list):
            tool_calls = [
                item
                for item in content
                if isinstance(item, dict) and item.get("type") == "tool_call"
            ]

    if not isinstance(tool_calls, list):
        return []

    normalized: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
            function = {"name": name, "arguments": arguments}

        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue

        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)

        normalized.append(
            {
                "id": tool_call.get("id") or f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
                "_index": index,
            }
        )
    return normalized


def _strip_internal_tool_call_fields(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in tool_call.items() if not k.startswith("_")} for tool_call in tool_calls]


def _sdk_message_reasoning_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""

    channels = message.get("channels")
    if not isinstance(channels, dict):
        return ""

    parts: list[str] = []
    for key in ("thought", "thinking", "reasoning", "analysis"):
        value = channels.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def _thinking_open_delta() -> dict[str, str]:
    return {"content": "<think>"}


def _thinking_delta(text: str) -> dict[str, str]:
    return {"content": text}


def _thinking_close_delta() -> dict[str, str]:
    return {"content": "</think>\n"}


def _is_context_overflow_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "input token ids are too long" in message
        or "exceeding the maximum number of tokens" in message
        or "max number of tokens reached" in message
    )


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
    settings = get_settings()
    profile_store = get_profile_store()

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
    bootstrap_system_prompt = profile_store.combined_bootstrap_system_prompt(message_dicts)
    effective_generation_params = profile_store.effective_generation_params(request)
    raw_tools = request.tools if settings.enable_tool_calling else None
    openai_tools = normalize_openai_tools(raw_tools)
    tool_sig = tools_signature(raw_tools)
    extra_context = _build_extra_context(settings, request)
    extra_context_sig = _json_signature(extra_context)

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
        bootstrap_system_message=bootstrap_system_prompt,
        initialized_with_profile=True,
        tools=openai_tools,
        tool_signature=tool_sig,
        automatic_tool_calling=False,
        extra_context=extra_context,
        extra_context_signature=extra_context_sig,
        filter_channel_content_from_kv_cache=settings.filter_thinking_from_kv_cache,
    )

    if effective_generation_params:
        logger.info(
            "Effective generation params for conversation %s: %s",
            conversation_id,
            sorted(effective_generation_params.keys()),
        )
    
    update_engine_activity()

    if request.stream:

        async def event_stream() -> AsyncIterator[str]:
            async with state.lock:
                state.touch()
                update_engine_activity()
                await manager.prepare_for_turn(state, incremental_payload)

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

                streamed_text_parts: list[str] = []
                streamed_tool_calls: list[dict[str, Any]] = []
                thinking_open = False
                completed = False

                for attempt in range(2):
                    try:
                        send_kwargs = _build_method_kwargs(
                            state.conversation.send_message_async,
                            effective_generation_params,
                        )
                        iterator = state.conversation.send_message_async(
                            incremental_payload,
                            **send_kwargs,
                        )

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
                                reasoning_piece = _sdk_message_reasoning_text(sdk_chunk)
                                if reasoning_piece:
                                    if not thinking_open:
                                        payload = {
                                            "id": completion_id,
                                            "object": "chat.completion.chunk",
                                            "created": created,
                                            "model": request.model,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": _thinking_open_delta(),
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                        yield _sse_data(payload)
                                        thinking_open = True
                                    payload = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": request.model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": _thinking_delta(reasoning_piece),
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                    yield _sse_data(payload)

                                tool_calls = _normalize_tool_calls(sdk_chunk)
                                if tool_calls:
                                    streamed_tool_calls.extend(tool_calls)
                                    payload = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": request.model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {
                                                    "tool_calls": [
                                                        {
                                                            "index": tool_call["_index"],
                                                            **_strip_internal_tool_call_fields([tool_call])[0],
                                                        }
                                                        for tool_call in tool_calls
                                                    ]
                                                },
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                    yield _sse_data(payload)
                                    continue

                                text_piece = sdk_message_to_text(sdk_chunk)
                                if not text_piece:
                                    continue
                                if thinking_open:
                                    payload = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": request.model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": _thinking_close_delta(),
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                    yield _sse_data(payload)
                                    thinking_open = False
                                streamed_text_parts.append(text_piece)

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

                        await manager.register_turn(
                            state,
                            incremental_payload,
                            "".join(streamed_text_parts),
                        )
                        completed = True
                        break
                            
                    except Exception as exc:
                        if attempt == 0 and _is_context_overflow_error(exc):
                            logger.warning(
                                "Streaming context overflow for conversation %s; recovering and retrying once",
                                conversation_id,
                            )
                            streamed_text_parts.clear()
                            streamed_tool_calls.clear()
                            await manager.recover_from_context_overflow(state)
                            continue

                        logger.exception("Streaming failed for conversation %s", conversation_id)
                        err_payload = {
                            "error": {
                                "message": str(exc),
                                "type": "internal_error",
                                "code": None,
                            }
                        }
                        yield _sse_data(err_payload)
                        break

                if thinking_open:
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": _thinking_close_delta(),
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield _sse_data(payload)

                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls" if streamed_tool_calls else "stop",
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
        for attempt in range(2):
            try:
                await manager.prepare_for_turn(state, incremental_payload)
                send_kwargs = _build_method_kwargs(
                    state.conversation.send_message,
                    effective_generation_params,
                )
                sdk_response = await asyncio.to_thread(
                    state.conversation.send_message,
                    incremental_payload,
                    **send_kwargs,
                )
                break
            except Exception as exc:
                if attempt == 0 and _is_context_overflow_error(exc):
                    logger.warning(
                        "Completion context overflow for conversation %s; recovering and retrying once",
                        conversation_id,
                    )
                    await manager.recover_from_context_overflow(state)
                    continue
                logger.exception("Completion failed for conversation %s", conversation_id)
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        turn_tool_calls = _normalize_tool_calls(sdk_response)
        await manager.register_turn(
            state,
            incremental_payload,
            "" if turn_tool_calls else sdk_message_to_text(sdk_response),
        )

    response_text = sdk_message_to_text(sdk_response)
    tool_calls = turn_tool_calls
    if tool_calls:
        message_payload: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": _strip_internal_tool_call_fields(tool_calls),
        }
        prompt_text = "\n".join(normalize_text_content(msg.get("content")) for msg in message_dicts)
        prompt_tokens = _estimate_token_count(prompt_text)
        return JSONResponse(content={
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": message_payload,
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": prompt_tokens,
            },
        })
    reasoning_text = _sdk_message_reasoning_text(sdk_response)

    prompt_text = "\n".join(normalize_text_content(msg.get("content")) for msg in message_dicts)
    prompt_tokens = _estimate_token_count(prompt_text)
    completion_tokens = _estimate_token_count(response_text)

    if reasoning_text:
        return JSONResponse(content={
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"<think>{reasoning_text}</think>\n{response_text}",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })

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
