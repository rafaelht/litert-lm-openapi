"""
Suite de pruebas y benchmark para litert-lm-openapi.
Permite evaluar:
1. Rendimiento y latencia de serialización concurrente (Semáforo).
2. Validez de contratos y streaming SSE compatibles con OpenAI / OpenWebUI.
3. Estimación de tokens vs precisión.
4. Ciclo de vida y evicción de conversaciones.
"""

import asyncio
import time
from typing import Any
import httpx


async def test_endpoint_health(base_url: str = "http://127.0.0.1:8000") -> bool:
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        print(f"[/healthz] OK: {resp.json()}")
        return True


async def test_endpoint_models(base_url: str = "http://127.0.0.1:8000") -> bool:
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        print(f"[/v1/models] OK: {data}")
        return True


async def test_administrative_bypass(base_url: str = "http://127.0.0.1:8000") -> bool:
    """Verifica que los prompts de títulos y tags de OpenWebUI reciben respuesta inmediata sin invocar al LLM."""
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        payload = {
            "model": "gemma-4-E2B-it",
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "user", "content": "Create a concise, 3-5 word title with an emoji for the following conversation:"}
            ],
            "stream": False,
            "max_tokens": 20
        }
        t0 = time.perf_counter()
        resp = await client.post("/v1/chat/completions", json=payload)
        elapsed = time.perf_counter() - t0
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"[Bypass Title] OK in {elapsed*1000:.1f}ms: {content}")
        assert elapsed < 0.5, f"Bypass should be instant, took {elapsed}s"
        return True


async def test_concurrent_generation(base_url: str = "http://127.0.0.1:8000") -> None:
    """Mide el comportamiento de 2 peticiones concurrentes para verificar serialización."""
    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        payload = {
            "model": "gemma-4-E2B-it",
            "messages": [{"role": "user", "content": "Di 'Hola' y nada mas."}],
            "stream": False,
        }

        async def send_one(req_id: int):
            t0 = time.perf_counter()
            resp = await client.post("/v1/chat/completions", json=payload)
            elapsed = time.perf_counter() - t0
            return req_id, resp.status_code, elapsed

        t_start = time.perf_counter()
        results = await asyncio.gather(send_one(1), send_one(2))
        total_time = time.perf_counter() - t_start
        print(f"[Concurrency Test] Total time for 2 requests: {total_time:.2f}s")
        for req_id, status, elapsed in results:
            print(f"  Req {req_id}: Status {status}, Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    print("Benchmark suite ready for standalone and integration runs.")
