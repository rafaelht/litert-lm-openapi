from __future__ import annotations

import inspect
import json
import logging
import time
from typing import Any, AsyncIterator

from anyio import to_thread
from fastapi import Request

from app.conversation_manager import ConversationManager, ConversationState
from app.engine import force_garbage_collection, update_engine_activity
from app.metrics import compute_usage_and_metrics
from app.utils import (
    extract_chunk_content_and_thought,
    extract_tool_calls_from_text,
    sdk_message_to_text,
)

logger = logging.getLogger(__name__)


def sse_data(payload: dict[str, Any] | str) -> str:
    """Formatea una carga de datos para el protocolo SSE de OpenAI."""
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_method_kwargs(method: Any, generation_params: dict[str, Any]) -> dict[str, Any]:
    """Filtra y adapta los parámetros de generación a la firma del método de LiteRT-LM."""
    if not generation_params:
        params: dict[str, Any] = {}
    else:
        params = dict(generation_params)

    result_kwargs: dict[str, Any] = {}

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return {}

    # 1. Adaptar max_output_tokens con límite de seguridad por turno
    max_tokens = params.get("max_tokens") or params.get("max_output_tokens")
    if not max_tokens or int(max_tokens) <= 0:
        # Límite por turno para evitar que respuestas fuera de control devoren la memoria RAM
        max_tokens = 1024
    if "max_output_tokens" in signature.parameters:
        result_kwargs["max_output_tokens"] = int(max_tokens)

    # 2. Configurar RepetitionPenaltyConfig para evitar bucles de repetición
    if "repetition_penalty_config" in signature.parameters:
        try:
            from litert_lm.interfaces import RepetitionPenaltyConfig
            rep_val = params.get("repetition_penalty", 1.15)
            pres_val = params.get("presence_penalty", 0.1)
            freq_val = params.get("frequency_penalty", 0.1)
            result_kwargs["repetition_penalty_config"] = RepetitionPenaltyConfig(
                repetition_penalty=float(rep_val) if rep_val is not None else 1.15,
                presence_penalty=float(pres_val) if pres_val is not None else 0.1,
                frequency_penalty=float(freq_val) if freq_val is not None else 0.1,
            )
        except Exception as e:
            logger.warning("No se pudo configurar RepetitionPenaltyConfig: %s", e)

    # 3. Configurar NoRepeatNgramConfig para evitar repeticiones exactas de n-gramas
    if "no_repeat_ngram_config" in signature.parameters:
        try:
            from litert_lm.interfaces import NoRepeatNgramConfig
            ngram_size = params.get("no_repeat_ngram_size", 4)
            result_kwargs["no_repeat_ngram_config"] = NoRepeatNgramConfig(
                no_repeat_ngram_size=int(ngram_size) if ngram_size is not None else 4
            )
        except Exception as e:
            logger.warning("No se pudo configurar NoRepeatNgramConfig: %s", e)

    # 4. Parámetros directos compatibles
    for key, value in params.items():
        if key in signature.parameters and key not in result_kwargs:
            result_kwargs[key] = value

    return result_kwargs


async def create_chat_event_stream(
    request: ChatCompletionRequest,
    raw_request: Request,
    state: ConversationState,
    manager: ConversationManager,
    incremental_payload: str | dict[str, Any],
    effective_generation_params: dict[str, Any],
    completion_id: str,
    created: int,
    thinking_override: bool,
    prompt_text: str,
) -> AsyncIterator[str]:
    """Generador de eventos Server-Sent Events (SSE) para chat streaming."""
    t_start = time.perf_counter()
    t_first_token: float | None = None
    streamed_text_parts: list[str] = []

    async with state.lock:
        state.touch()
        update_engine_activity()
        await manager.prepare_for_turn(
            state,
            incremental_payload,
            thinking_enabled=bool(thinking_override),
        )

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
        yield sse_data(first_chunk)

        try:
            send_kwargs = build_method_kwargs(
                state.conversation.send_message_async,
                effective_generation_params,
            )
            iterator = state.conversation.send_message_async(
                incremental_payload,
                **send_kwargs,
            )

            yielded_len = 0
            tool_tag_prefix = "<tool_call>"

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
                    content_piece, thought_piece = extract_chunk_content_and_thought(sdk_chunk)

                    # 1. Si el modelo emitió razonamiento (thought/reasoning), emitirlo como reasoning_content
                    if thought_piece:
                        thought_payload = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "reasoning_content": thought_piece,
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield sse_data(thought_payload)

                    # 2. Si hay texto normal de respuesta, procesarlo y emitirlo en content
                    if not content_piece:
                        continue

                    streamed_text_parts.append(content_piece)
                    accumulated = "".join(streamed_text_parts)

                    # Si hay una tool_call en progreso, retener el texto de la tool_call
                    if "<tool_call>" in accumulated:
                        before_tool = accumulated.split("<tool_call>", 1)[0]
                        if len(before_tool) > yielded_len:
                            delta_text = before_tool[yielded_len:]
                            yielded_len = len(before_tool)
                            payload = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}],
                            }
                            yield sse_data(payload)
                    else:
                        # Comprobar si el final coincide con prefijo de '<tool_call>'
                        tail_len = 0
                        for i in range(len(tool_tag_prefix) - 1, 0, -1):
                            if accumulated.endswith(tool_tag_prefix[:i]):
                                tail_len = i
                                break

                        safe_end = len(accumulated) - tail_len
                        if safe_end > yielded_len:
                            delta_text = accumulated[yielded_len:safe_end]
                            yielded_len = safe_end
                            payload = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}],
                            }
                            yield sse_data(payload)

            # Al finalizar el stream, emitir texto restante si no es tool_call
            accumulated = "".join(streamed_text_parts)
            if "<tool_call>" not in accumulated and len(accumulated) > yielded_len:
                delta_text = accumulated[yielded_len:]
                yielded_len = len(accumulated)
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}],
                }
                yield sse_data(payload)

            t_end = time.perf_counter()
            full_response = accumulated
            await manager.register_turn(
                state,
                incremental_payload,
                full_response,
            )

        except Exception as exc:
            t_end = time.perf_counter()
            logger.exception("Streaming failed for conversation %s", state.conversation_id)
            err_payload = {
                "error": {
                    "message": str(exc),
                    "type": "internal_error",
                    "code": None,
                }
            }
            yield sse_data(err_payload)

        full_response = "".join(streamed_text_parts)
        usage_dict = compute_usage_and_metrics(
            state.conversation,
            prompt_text,
            full_response,
            t_start,
            t_first_token,
            t_end,
        )

        extracted_tool_calls = extract_tool_calls_from_text(full_response)
        finish_reason = "tool_calls" if extracted_tool_calls else "stop"

        if extracted_tool_calls:
            logger.info("[TOOLS] Despachando llamada a tool para OpenWebUI: %s", extracted_tool_calls)
            tool_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": extracted_tool_calls},
                        "finish_reason": None,
                    }
                ],
            }
            yield sse_data(tool_chunk)

        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage_dict,
            "prompt_eval_count": usage_dict.get("prompt_eval_count"),
            "prompt_eval_duration": usage_dict.get("prompt_eval_duration"),
            "eval_count": usage_dict.get("eval_count"),
            "eval_duration": usage_dict.get("eval_duration"),
            "total_duration": usage_dict.get("total_duration"),
        }
        yield sse_data(final_chunk)

        # Chunk de uso compatible con OpenAI stream_options
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
        yield sse_data(usage_chunk)
        yield sse_data("[DONE]")
        force_garbage_collection()
