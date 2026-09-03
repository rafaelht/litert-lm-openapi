from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.admin_handlers import detect_admin_request, handle_admin_completion
from app.config import get_settings
from app.conversation_manager import get_conversation_manager
from app.engine import (
    check_and_consume_reload_flag,
    force_garbage_collection,
    init_engine,
    update_engine_activity,
)
from app.metrics import compute_usage_and_metrics
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
from app.streaming import build_method_kwargs, create_chat_event_stream
from app.utils import (
    bootstrap_messages,
    extract_api_key,
    extract_incremental_message_payload,
    extract_tool_calls_from_text,
    format_tools_system_prompt,
    make_conversation_id,
    normalize_text_content,
    sdk_message_to_text,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai-compatible"])


def _is_thinking_requested(request: ChatCompletionRequest) -> bool:
    """Verifica si OpenWebUI o el cliente solicitó razonamiento/thinking en los parámetros."""
    if request.thinking is not None:
        if isinstance(request.thinking, bool):
            return request.thinking
        if isinstance(request.thinking, dict):
            return request.thinking.get("type") in {"enabled", "true", True} or bool(request.thinking.get("enabled"))
        if isinstance(request.thinking, str):
            return request.thinking.lower() in {"true", "enabled", "on", "1"}

    if request.reasoning_effort is not None:
        return str(request.reasoning_effort).lower() not in {"none", "off", "0", "false", ""}

    if request.model_extra:
        for key in ["thinking", "thought", "reasoning", "enable_thinking"]:
            val = request.model_extra.get(key)
            if val is True or val == "true" or val == "True":
                return True
            if isinstance(val, str) and val.lower() in {"true", "enabled", "on", "1"}:
                return True
            if isinstance(val, dict) and (val.get("type") in {"enabled", "true"} or val.get("enabled") is True):
                return True

        if "reasoning_effort" in request.model_extra:
            effort = str(request.model_extra["reasoning_effort"]).lower()
            if effort not in {"none", "off", "0", "false", ""}:
                return True

        for container_key in ["extra_body", "chat_options", "options", "custom_params", "params"]:
            container = request.model_extra.get(container_key)
            if isinstance(container, dict):
                for key in ["thinking", "thought", "reasoning", "enable_thinking"]:
                    val = container.get(key)
                    if val is True or val == "true" or val == "True":
                        return True
                    if isinstance(val, str) and val.lower() in {"true", "enabled", "on", "1"}:
                        return True
                    if isinstance(val, dict) and (val.get("type") in {"enabled", "true"} or val.get("enabled") is True):
                        return True
                if "reasoning_effort" in container:
                    effort = str(container["reasoning_effort"]).lower()
                    if effort not in {"none", "off", "0", "false", ""}:
                        return True

    return False


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

    incremental_payload = extract_incremental_message_payload(message_dicts)
    if isinstance(incremental_payload, str):
        incremental_message = incremental_payload.strip()
    else:
        incremental_message = normalize_text_content(incremental_payload.get("content", "")).strip()

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # 1. Cortocircuito para peticiones administrativas de OpenWebUI (Títulos y Tags)
    is_admin, admin_type = detect_admin_request(
        message_dicts=message_dicts,
        incremental_message=incremental_message,
        max_tokens=request.max_tokens,
    )
    if is_admin:
        return await handle_admin_completion(
            request=request,
            req_type=admin_type,
            message_dicts=message_dicts,
            incremental_message=incremental_message,
            created=created,
            completion_id=completion_id,
        )

    # 2. Flujo normal de conversación
    engine_instance = await init_engine()

    api_key = extract_api_key(authorization)
    conversation_id = make_conversation_id(api_key, request.model, message_dicts)
    manager = get_conversation_manager()

    thinking_override = _is_thinking_requested(request)
    if thinking_override:
        logger.info("[THINKING] Modo thinking activado para conversación %s", conversation_id)

    tools_system_prompt = format_tools_system_prompt(request.tools)
    if tools_system_prompt:
        logger.info("[TOOLS] %d herramientas inyectadas en la conversación %s", len(request.tools), conversation_id)

    bootstrap_system_prompt = profile_store.combined_bootstrap_system_prompt(
        message_dicts,
        thinking_override=thinking_override,
    )
    if tools_system_prompt:
        bootstrap_system_prompt = (
            f"{bootstrap_system_prompt}\n\n{tools_system_prompt}"
            if bootstrap_system_prompt
            else tools_system_prompt
        )

    effective_generation_params = profile_store.effective_generation_params(request)

    # Si el motor se recreó, actualizar referencias internas y limpiar estado previo
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
        thinking_enabled=bool(thinking_override),
    )

    update_engine_activity()
    prompt_text = "\n".join(normalize_text_content(msg.get("content")) for msg in message_dicts)

    # Streaming SSE
    if request.stream:
        return StreamingResponse(
            create_chat_event_stream(
                request=request,
                raw_request=raw_request,
                state=state,
                manager=manager,
                incremental_payload=incremental_payload,
                effective_generation_params=effective_generation_params,
                completion_id=completion_id,
                created=created,
                thinking_override=thinking_override,
                prompt_text=prompt_text,
            ),
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
            await manager.prepare_for_turn(
                state,
                incremental_payload,
                thinking_enabled=bool(thinking_override),
            )
            send_kwargs = build_method_kwargs(
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

    usage_dict = compute_usage_and_metrics(
        state.conversation,
        prompt_text,
        response_text,
        t_start,
        None,
        t_end,
    )

    extracted_tool_calls = extract_tool_calls_from_text(response_text)
    finish_reason = "tool_calls" if extracted_tool_calls else "stop"

    response = ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(
                    content=response_text if not extracted_tool_calls else None,
                    tool_calls=extracted_tool_calls,
                ),
                finish_reason=finish_reason,
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