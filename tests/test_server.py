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
        self.assertEqual(settings.max_active_conversations, 1)
        self.assertEqual(settings.thinking_token_budget, 384)
        self.assertEqual(settings.context_rollover_threshold_tokens, 2600)
        self.assertEqual(settings.context_rollover_headroom_tokens, 1500)
        self.assertEqual(settings.engine_ttl, 0)
        self.assertFalse(settings.enable_thinking)
        self.assertFalse(settings.enable_tools)
        self.assertFalse(settings.force_static_max_tokens)

    def test_heuristic_title_generation(self):
        # Texto limpio estándar
        title1 = generate_heuristic_title("¿Cómo funciona el motor de un avión comercial?")
        self.assertIn("Cómo", title1)
        self.assertTrue(len(title1) <= 35)

        # Plantilla compleja de OpenWebUI con Task, Chat, User y Assistant
        openwebui_prompt = (
            "### Task:\n"
            "Generate a short 3-5 word title for the following conversation:\n"
            "### Chat:\n"
            "User: ¿Cómo funciona un avión?\n"
            "Assistant: Un avión funciona gracias a la sustentación"
        )
        title_owui = generate_heuristic_title(openwebui_prompt)
        self.assertNotIn("task", title_owui.lower())
        self.assertNotIn("###", title_owui)
        self.assertIn("Cómo funciona un avión", title_owui)

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

            self.assertEqual(state.rollover_count, 1)
            self.assertEqual(state.conversation, mock_conv_new)
            self.assertTrue(len(state.summary_text) > 0)
            # Los mensajes recientes deben conservarse dentro del presupuesto de pares
            self.assertTrue(len(state.rolling_messages) <= 8)
            self.assertTrue(mock_conv_old.close.called)
        finally:
            loop.close()

    def test_thinking_and_reasoning_handling(self):
        from app.utils import extract_chunk_content_and_thought, sdk_message_to_text

        # 1. Fragmento de pensamiento serializado como string JSON (el caso del reporte del usuario)
        raw_user_chunk = (
            '{"role": "assistant", "channels": {"thought": "Thinking Process:"}, "reasoning_content": "Thinking Process:"}'
        )
        content, thought = extract_chunk_content_and_thought(raw_user_chunk)
        self.assertEqual(content, "")
        self.assertEqual(thought, "Thinking Process:")

        # sdk_message_to_text debe descartar el JSON y devolver vacío para no contaminar el chat
        self.assertEqual(sdk_message_to_text(raw_user_chunk), "")

        # 2. Fragmento de diccionario con channels
        dict_chunk = {
            "role": "assistant",
            "channels": {"thought": "Analizando"},
            "reasoning_content": "Analizando",
        }
        content2, thought2 = extract_chunk_content_and_thought(dict_chunk)
        self.assertEqual(content2, "")
        self.assertEqual(thought2, "Analizando")
        self.assertEqual(sdk_message_to_text(dict_chunk), "")

        # 3. Fragmento de respuesta normal final
        final_chunk = "Un avión funciona gracias a la sustentación"
        content3, thought3 = extract_chunk_content_and_thought(final_chunk)
        self.assertEqual(content3, "Un avión funciona gracias a la sustentación")
        self.assertEqual(thought3, "")
        self.assertEqual(sdk_message_to_text(final_chunk), "Un avión funciona gracias a la sustentación")

    def test_is_thinking_requested(self):
        from unittest.mock import patch
        from app.config import Settings
        from app.openai_routes import _is_thinking_requested
        from app.schemas import ChatCompletionRequest, ChatMessage

        req_default = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hola")],
            reasoning_effort="default",
        )
        req_thinking_true = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hola")],
            thinking=True,
        )
        req_high = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hola")],
            reasoning_effort="high",
        )

        # 1. Con ENABLE_THINKING=false (por defecto en .env para máxima velocidad)
        # El thinking está 100% apagado y cualquier parámetro del cliente se ignora.
        self.assertFalse(_is_thinking_requested(req_default))
        self.assertFalse(_is_thinking_requested(req_thinking_true))
        self.assertFalse(_is_thinking_requested(req_high))

        # 2. Si el usuario activa ENABLE_THINKING=true en .env en el futuro:
        with patch("app.openai_routes.get_settings") as mock_settings:
            mock_settings.return_value.enable_thinking = True
            self.assertFalse(_is_thinking_requested(req_default))
            self.assertTrue(_is_thinking_requested(req_thinking_true))
            self.assertTrue(_is_thinking_requested(req_high))


if __name__ == "__main__":
    unittest.main()
