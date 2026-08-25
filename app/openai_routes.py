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
from app.engine import get_engine, init_engine, update_engine_activity, check_and_consume_reload_flag, force_garbage_collection
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


def _compute_usage_and_metrics(
    conversation: Any,
    prompt_text: str,
    response_text: str,
    t_start: float,
    t_first_token: float | None,
    t_end: float,
) -> dict[str, Any]:
    prompt_tokens = _estimate_token_count(prompt_text)
    completion_tokens = _estimate_token_count(response_text)

    bench = None
    try:
        if hasattr(conversation, "get_benchmark_info"):
            bench = conversation.get_benchmark_info()
    except Exception:
        bench = None

    total_duration_ns = int(max(0.001, t_end - t_start) * 1e9)
    load_duration_ns = int(bench.init_time_in_second * 1e9) if bench else 0

    if bench and bench.last_prefill_token_count > 0:
        p_tokens = bench.last_prefill_token_count
        p_duration_ns = int((p_tokens / max(0.1, bench.last_prefill_tokens_per_second)) * 1e9) if bench.last_prefill_tokens_per_second > 0 else 0
    else:
        p_tokens = prompt_tokens
        if t_first_token is not None and t_first_token > t_start:
            p_duration_ns = int((t_first_token - t_start) * 1e9)
        else:
            p_duration_ns = 0

    if bench and bench.last_decode_token_count > 0:
        c_tokens = bench.last_decode_token_count
        c_duration_ns = int((c_tokens / max(0.1, bench.last_decode_tokens_per_second)) * 1e9) if bench.last_decode_tokens_per_second > 0 else 0
    else:
        c_tokens = completion_tokens
        if t_first_token is not None:
            c_duration_ns = int(max(0.001, t_end - t_first_token) * 1e9)
        else:
            c_duration_ns = total_duration_ns

    final_prompt_tokens = prompt_tokens if prompt_tokens > 0 else p_tokens
    final_completion_tokens = completion_tokens if completion_tokens > 0 else c_tokens

    return {
        "prompt_tokens": final_prompt_tokens,
        "completion_tokens": final_completion_tokens,
        "total_tokens": final_prompt_tokens + final_completion_tokens,
        "prompt_eval_count": p_tokens,
        "prompt_eval_duration": p_duration_ns,
        "eval_count": c_tokens,
        "eval_duration": c_duration_ns,
        "total_duration": total_duration_ns,
        "load_duration": load_duration_ns,
    }


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
    )

    if effective_generation_params:
        logger.info(
            "Effective generation params for conversation %s: %s",
            conversation_id,
            sorted(effective_generation_params.keys()),
        )
    
    update_engine_activity()

    prompt_text = "\n".join(normalize_text_content(msg.get("content")) for msg in message_dicts)

    if request.stream:

        async def event_stream() -> AsyncIterator[str]:
            t_start = time.perf_counter()
            t_first_token: float | None = None
            streamed_text_parts: list[str] = []

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

                        if t_first_token is None:
                            t_first_token = time.perf_counter()

                        state.touch()
                        update_engine_activity()
                        
                        if not disconnected:
                            text_piece = sdk_message_to_text(sdk_chunk)
                            if not text_piece:
                                continue
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

                    t_end = time.perf_counter()
                    full_response = "".join(streamed_text_parts)
                    await manager.register_turn(
                        state,
                        incremental_payload,
                        full_response,
                    )
                        
                except Exception as exc:
                    t_end = time.perf_counter()
                    logger.exception("Streaming failed for conversation %s", conversation_id)
                    err_payload = {
                        "error": {
                            "message": str(exc),
                            "type": "internal_error",
                            "code": None,
                        }
                    }
                    yield _sse_data(err_payload)

                full_response = "".join(streamed_text_parts)
                usage_dict = _compute_usage_and_metrics(
                    state.conversation,
                    prompt_text,
                    full_response,
                    t_start,
                    t_first_token,
                    t_end,
                )

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
                    "usage": usage_dict,
                    "prompt_eval_count": usage_dict.get("prompt_eval_count"),
                    "prompt_eval_duration": usage_dict.get("prompt_eval_duration"),
                    "eval_count": usage_dict.get("eval_count"),
                    "eval_duration": usage_dict.get("eval_duration"),
                    "total_duration": usage_dict.get("total_duration"),
                }
                yield _sse_data(final_chunk)

                # Chunk explícito de uso (OpenAI stream_options)
                usage_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [],
                    "usage": usage_dict,
                    "prompt_eval_count": usage_dict.get("prompt_eval_count"),
                    "prompt_eval_duration": usage_dict.get("prompt_eval_duration"),
                    "eval_count": usage_dict.get("eval_count"),
                    "eval_duration": usage_dict.get("eval_duration"),
                    "total_duration": usage_dict.get("total_duration"),
                }
                yield _sse_data(usage_chunk)
                yield _sse_data("[DONE]")
                force_garbage_collection()

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
    t_start = time.perf_counter()
    async with state.lock:
        state.touch()
        update_engine_activity()
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
        except Exception as exc:
            logger.exception("Completion failed for conversation %s", conversation_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        
        t_end = time.perf_counter()
        response_text = sdk_message_to_text(sdk_response)
        await manager.register_turn(
            state,
            incremental_payload,
            response_text,
        )

    usage_dict = _compute_usage_and_metrics(
        state.conversation,
        prompt_text,
        response_text,
        t_start,
        None,
        t_end,
    )

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
        usage=ChatCompletionUsage(**usage_dict),
        prompt_eval_count=usage_dict.get("prompt_eval_count"),
        prompt_eval_duration=usage_dict.get("prompt_eval_duration"),
        eval_count=usage_dict.get("eval_count"),
        eval_duration=usage_dict.get("eval_duration"),
        total_duration=usage_dict.get("total_duration"),
        load_duration=usage_dict.get("load_duration"),
    )
    force_garbage_collection()

    return JSONResponse(content=response.model_dump(by_alias=True, exclude_none=True))