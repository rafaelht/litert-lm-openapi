from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from app.admin_handlers import (
    detect_admin_request,
    generate_heuristic_tags,
    generate_heuristic_title,
)
from app.config import Settings, get_settings
from app.metrics import compute_usage_and_metrics
from app.streaming import build_method_kwargs


class TestServerOptimizations(unittest.TestCase):
    def test_settings_defaults(self):
        settings = get_settings()
        self.assertEqual(settings.cpu_threads, 4)
        self.assertEqual(settings.max_num_tokens, 4096)
        self.assertFalse(settings.enable_xnnpack_cache)
        self.assertTrue(settings.use_ringbuffers_local_attention)
        self.assertFalse(settings.enable_benchmark)
        self.assertFalse(settings.enable_admin_llm)
        self.assertEqual(settings.max_active_conversations, 5)
        self.assertEqual(settings.thinking_token_budget, 384)
        self.assertEqual(settings.context_rollover_threshold_tokens, 3400)

    def test_heuristic_title_generation(self):
        # Texto limpio estándar
        title1 = generate_heuristic_title("¿Cómo funciona el motor de un avión comercial?")
        self.assertIn("Cómo", title1)
        self.assertTrue(len(title1) <= 35)

        # Prompt con prefijo administrativo
        title2 = generate_heuristic_title("Task: explain quantum computing")
        self.assertTrue(title2.startswith("Explain") or "quantum" in title2.lower())

        # Prompt vacío
        title3 = generate_heuristic_title("")
        self.assertEqual(title3, "Conversación General")

    def test_heuristic_tags_generation(self):
        tags_code = generate_heuristic_tags("Escribe una API en FastAPI con Python y Docker")
        self.assertIn("Programación", tags_code)
        self.assertIn("Sistemas", tags_code)

        tags_general = generate_heuristic_tags("Hola, buenos días")
        self.assertEqual(tags_general, ["General"])

    def test_detect_admin_request(self):
        # OpenWebUI petición de título
        is_admin, admin_type = detect_admin_request(
            message_dicts=[{"role": "user", "content": "Create a creative title for this chat"}],
            incremental_message="Create a creative title for this chat",
            max_tokens=20,
        )
        self.assertTrue(is_admin)
        self.assertEqual(admin_type, "title")

        # OpenWebUI petición de tags
        is_admin, admin_type = detect_admin_request(
            message_dicts=[{"role": "user", "content": "Generate 1-3 broad tags"}],
            incremental_message="Generate 1-3 broad tags",
            max_tokens=15,
        )
        self.assertTrue(is_admin)
        self.assertEqual(admin_type, "tags")

        # Conversación normal de usuario
        is_admin, admin_type = detect_admin_request(
            message_dicts=[{"role": "user", "content": "Explícame cómo vuelan los aviones"}],
            incremental_message="Explícame cómo vuelan los aviones",
            max_tokens=1000,
        )
        self.assertFalse(is_admin)
        self.assertEqual(admin_type, "")

    def test_build_method_kwargs(self):
        def sample_method(max_output_tokens: int = 100, temperature: float = 0.7):
            pass

        params = {"max_tokens": 256, "temperature": 0.5, "unsupported_param": "ignore"}
        kwargs = build_method_kwargs(sample_method, params)

        self.assertEqual(kwargs.get("max_output_tokens"), 256)
        self.assertEqual(kwargs.get("temperature"), 0.5)
        self.assertNotIn("unsupported_param", kwargs)

    def test_compute_usage_and_metrics(self):
        mock_conv = MagicMock()
        del mock_conv.get_benchmark_info  # Sin benchmark de C++

        metrics = compute_usage_and_metrics(
            conversation=mock_conv,
            prompt_text="Hola mundo",
            response_text="Hola, ¿en qué puedo ayudarte hoy?",
            t_start=1.0,
            t_first_token=1.2,
            t_end=2.0,
        )

        self.assertIn("prompt_tokens", metrics)
        self.assertIn("completion_tokens", metrics)
        self.assertIn("tokens_per_second", metrics)
        self.assertIn("eval_rate", metrics)
        self.assertTrue(metrics["tokens_per_second"] > 0)

    def test_handle_admin_completion_sync(self):
        import asyncio
        from app.admin_handlers import handle_admin_completion
        from app.schemas import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            model="gemma-test",
            messages=[ChatMessage(role="user", content="¿Cómo vuela un avión?")],
            stream=False,
        )

        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(
                handle_admin_completion(
                    request=req,
                    req_type="title",
                    message_dicts=[{"role": "user", "content": "¿Cómo vuela un avión?"}],
                    incremental_message="¿Cómo vuela un avión?",
                    created=123456789,
                    completion_id="chatcmpl-test",
                )
            )
            import json
            data = json.loads(resp.body.decode("utf-8"))
            self.assertEqual(data["id"], "chatcmpl-test")
            self.assertIn("choices", data)
            content = json.loads(data["choices"][0]["message"]["content"])
            self.assertIn("title", content)
            self.assertIn("Cómo", content["title"])
        finally:
            loop.close()

    def test_fastapi_endpoints(self):
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            # 1. Health check
            res_health = client.get("/healthz")
            self.assertEqual(res_health.status_code, 200)
            self.assertEqual(res_health.json(), {"status": "ok"})

            # 2. Models list
            res_models = client.get("/v1/models")
            self.assertEqual(res_models.status_code, 200)
            models_data = res_models.json()
            self.assertEqual(models_data["object"], "list")
            self.assertTrue(len(models_data["data"]) > 0)

            # 3. Intercepción administrativa de Título (OpenWebUI)
            res_title = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemma-test",
                    "messages": [{"role": "user", "content": "Create a creative title with an emoji"}],
                    "max_tokens": 20,
                    "stream": False,
                },
            )
            self.assertEqual(res_title.status_code, 200)
            title_json = json.loads(res_title.json()["choices"][0]["message"]["content"])
            self.assertIn("title", title_json)

            # 4. Intercepción administrativa de Título streaming (OpenWebUI)
            res_title_stream = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemma-test",
                    "messages": [{"role": "user", "content": "Create a creative title with an emoji"}],
                    "max_tokens": 20,
                    "stream": True,
                },
            )
            self.assertEqual(res_title_stream.status_code, 200)
            self.assertIn("data: ", res_title_stream.text)
            self.assertIn("[DONE]", res_title_stream.text)

    def test_context_rollover_and_summary(self):
        import asyncio
        from unittest.mock import AsyncMock
        from app.conversation_manager import ConversationManager, ConversationState

        mock_engine = MagicMock()
        mock_conv_old = MagicMock()
        mock_conv_old.token_count = 3600  # Casi al límite de 4096
        mock_conv_new = MagicMock()
        mock_conv_new.token_count = 300   # Fresco tras rollover

        mock_engine.create_conversation.return_value = mock_conv_new
        mock_engine.tokenize.return_value = [1] * 50

        manager = ConversationManager(mock_engine)

        state = ConversationState(
            conversation_id="conv-4096-test",
            conversation=mock_conv_old,
            bootstrap_system_message="Eres un asistente experto.",
            rolling_messages=[
                {"role": "user", "content": "Pregunta 1: Arquitectura de computadores"},
                {"role": "assistant", "content": "Respuesta 1: Explicación de CPU y buses."},
                {"role": "user", "content": "Pregunta 2: ¿Cómo funciona la memoria RAM DDR5?"},
                {"role": "assistant", "content": "Respuesta 2: Memoria de canal único y ancho de banda."},
                {"role": "user", "content": "Pregunta 3: ¿Qué es el KV-cache en LiteRT?"},
                {"role": "assistant", "content": "Respuesta 3: Almacenamiento de claves y valores."},
                {"role": "user", "content": "Pregunta 4: ¿Cómo funcionan los transformadores?"},
                {"role": "assistant", "content": "Respuesta 4: Mecanismo de auto-atención y decodificación."},
                {"role": "user", "content": "Pregunta 5: ¿Por qué limitar los tokens a 4096?"},
                {"role": "assistant", "content": "Respuesta 5: Para optimizar el tamaño de la ventana de contexto."},
            ],
        )

        loop = asyncio.new_event_loop()
        try:
            # Comprobar resumen estructurado
            summary = manager._build_structured_summary(
                older_messages=state.rolling_messages[:4],
                previous_summary="Introducción previa",
            )
            self.assertIn("Introducción previa", summary)
            self.assertIn("Arquitectura de computadores", summary)
            self.assertIn("memoria RAM DDR5", summary)

            # Probar prepare_for_turn cuando supera el umbral hacia 4096
            loop.run_until_complete(
                manager.prepare_for_turn(
                    state=state,
                    incoming_payload="Pregunta 4: Continúa con los detalles",
                    thinking_enabled=False,
                )
            )

            # Debe haber disparado el rollover
            self.assertEqual(state.rollover_count, 1)
            self.assertEqual(state.conversation, mock_conv_new)
            self.assertTrue(len(state.summary_text) > 0)
            # Los mensajes recientes deben conservarse dentro del presupuesto de pares
            self.assertTrue(len(state.rolling_messages) <= 8)
            self.assertTrue(mock_conv_old.close.called)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
