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

    # Cálculo explícito de velocidad de tokens
    eval_sec = max(0.001, c_duration_ns / 1e9)
    tokens_per_sec = round(final_completion_tokens / eval_sec, 2)
    prompt_sec = max(0.001, p_duration_ns / 1e9)
    prompt_tokens_per_sec = round(final_prompt_tokens / prompt_sec, 2)

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
        "tokens_per_second": tokens_per_sec,
        "eval_rate": f"{tokens_per_sec} tokens/s",
        "prompt_eval_rate": f"{prompt_tokens_per_sec} tokens/s",
    }


def _extract_title_from_response(raw_text: str) -> str:
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
    
    cleaned = re.sub(r'```(?:json)?|```', '', raw_text).strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        first = lines[0].replace('"', '').replace('{', '').replace('}', '').replace('title:', '').strip()
        return first if first else "Conversación General"
    return "Conversación General"


def _is_thinking_requested(request: ChatCompletionRequest) -> bool:
    """Verifica si OpenWebUI o el cliente solicitó razonamiento/thinking en los parámetros."""
    # 1. Parámetro directo 'thinking'
    if request.thinking is not None:
        if isinstance(request.thinking, bool):
            return request.thinking
        if isinstance(request.thinking, dict):
            return request.thinking.get("type") in {"enabled", "true", True} or bool(request.thinking.get("enabled"))
        if isinstance(request.thinking, str):
            return request.thinking.lower() in {"true", "enabled", "on", "1"}

    # 2. Parámetro 'reasoning_effort'
    if request.reasoning_effort is not None:
        return str(request.reasoning_effort).lower() not in {"none", "off", "0", "false", ""}

    # 3. Parámetros extra enviados por OpenWebUI / extensiones
    if request.model_extra:
        for key in ["thinking", "thought", "reasoning", "enable_thinking"]:
            val = request.model_extra.get(key)
            if val is True:
                return True
            if isinstance(val, str) and val.lower() in {"true", "enabled", "on", "1"}:
                return True
            if isinstance(val, dict) and (val.get("type") in {"enabled", "true"} or val.get("enabled") is True):
                return True
        if "reasoning_effort" in request.model_extra:
            effort = str(request.model_extra["reasoning_effort"]).lower()
            if effort not in {"none", "off", "0", "false", ""}:
                return True
        if "chat_options" in request.model_extra and isinstance(request.model_extra["chat_options"], dict):
            if request.model_extra["chat_options"].get("thinking") is True:
                return True

    return False


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
        logger.info("[TITLE/TAGS] Procesando petición de título/tags de OpenWebUI.")
        
        if is_title_req:
            chat_title = "Conversación General"
            try:
                engine = await init_engine()
                title_conv = engine.create_conversation(
                    system_message='Eres un asistente que resume conversaciones. Genera un título muy conciso y creativo de 3 a 5 palabras con un emoji alusivo en formato JSON: {"title": "..."}.',
                    max_output_tokens=25,
                )
                title_response = await asyncio.to_thread(
                    title_conv.send_message,
                    incremental_message,
                )
                title_conv.close()
                raw_text = sdk_message_to_text(title_response)
                chat_title = _extract_title_from_response(raw_text)
                logger.info("[TITLE] Título generado por IA: %s", chat_title)
            except Exception as e:
                logger.warning("[TITLE ERROR] Falló generación de título por IA, usando heurística: %s", str(e))
                chat_title = _generate_heuristic_title(incremental_message)
            
            mock_payload = {"title": chat_title}

        else:
            try:
                engine = await init_engine()
                tags_conv = engine.create_conversation(
                    system_message='Genera 1 a 3 etiquetas muy breves en formato lista JSON: ["tag1", "tag2"].',
                    max_output_tokens=20,
                )
                tags_response = await asyncio.to_thread(
                    tags_conv.send_message,
                    incremental_message,
                )
                tags_conv.close()
                raw_text = sdk_message_to_text(tags_response)
                try:
                    mock_payload = json.loads(raw_text)
                    if not isinstance(mock_payload, list):
                        mock_payload = [str(mock_payload)]
                except Exception:
                    mock_payload = ["General"]
            except Exception as e:
                logger.warning("[TAGS ERROR] Falló generación de tags: %s", str(e))
                mock_payload = ["General"]

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
    
    thinking_override = _is_thinking_requested(request)
    bootstrap_system_prompt = profile_store.combined_bootstrap_system_prompt(
        message_dicts,
        thinking_override=thinking_override,
    )
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