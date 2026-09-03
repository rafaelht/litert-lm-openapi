from __future__ import annotations

import logging
from typing import Any

from app.engine import get_engine

logger = logging.getLogger(__name__)


def estimate_token_count(text: str) -> int:
    """Calcula la cantidad de tokens de un texto usando el tokenizer del motor."""
    if not text:
        return 0

    try:
        engine = get_engine()
        if engine is not None:
            tokens = engine.tokenize(text)
            if isinstance(tokens, list) and tokens:
                return len(tokens)
    except Exception:
        pass

    # Estimación heurística de tokens si el motor aún no está activo
    return max(1, len(text.split()))


def compute_usage_and_metrics(
    conversation: Any,
    prompt_text: str,
    response_text: str,
    t_start: float,
    t_first_token: float | None,
    t_end: float,
) -> dict[str, Any]:
    """Calcula las métricas de uso compatibles con OpenAI y estadísticas de velocidad."""
    prompt_tokens = estimate_token_count(prompt_text)
    completion_tokens = estimate_token_count(response_text)

    bench = None
    try:
        if hasattr(conversation, "get_benchmark_info"):
            bench = conversation.get_benchmark_info()
    except Exception:
        bench = None

    total_duration_ns = int(max(0.001, t_end - t_start) * 1e9)
    load_duration_ns = int(bench.init_time_in_second * 1e9) if bench else 0

    if bench and hasattr(bench, "last_prefill_token_count") and bench.last_prefill_token_count > 0:
        p_tokens = bench.last_prefill_token_count
        p_duration_ns = (
            int((p_tokens / max(0.1, bench.last_prefill_tokens_per_second)) * 1e9)
            if bench.last_prefill_tokens_per_second > 0
            else 0
        )
    else:
        p_tokens = prompt_tokens
        if t_first_token is not None and t_first_token > t_start:
            p_duration_ns = int((t_first_token - t_start) * 1e9)
        else:
            p_duration_ns = 0

    if bench and hasattr(bench, "last_decode_token_count") and bench.last_decode_token_count > 0:
        c_tokens = bench.last_decode_token_count
        c_duration_ns = (
            int((c_tokens / max(0.1, bench.last_decode_tokens_per_second)) * 1e9)
            if bench.last_decode_tokens_per_second > 0
            else 0
        )
    else:
        c_tokens = completion_tokens
        if t_first_token is not None:
            c_duration_ns = int(max(0.001, t_end - t_first_token) * 1e9)
        else:
            c_duration_ns = total_duration_ns

    final_prompt_tokens = prompt_tokens if prompt_tokens > 0 else p_tokens
    final_completion_tokens = completion_tokens if completion_tokens > 0 else c_tokens

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
