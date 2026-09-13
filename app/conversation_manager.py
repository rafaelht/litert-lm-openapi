from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

from litert_lm import Conversation, Engine

from app.config import get_settings
from app.engine import force_garbage_collection
from app.utils import normalize_text_content, now_ts, sdk_message_to_text

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    conversation_id: str
    conversation: Conversation
    bootstrap_system_message: str = ""
    rolling_messages: list[dict[str, Any]] = field(default_factory=list)
    summary_text: str = ""
    last_known_token_count: int = 0
    rollover_count: int = 0
    initialized_with_profile: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_access: float = field(default_factory=now_ts)

    def touch(self) -> None:
        self.last_access = now_ts()


class ConversationManager:
    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine
        self._settings = get_settings()
        self._conversations: dict[str, ConversationState] = {}
        self._manager_lock = asyncio.Lock()
        self._rollover_threshold_tokens = max(1, self._settings.context_rollover_threshold_tokens)
        configured_recent = self._settings.context_rollover_recent_messages
        self._rollover_recent_messages = min(10, max(1, configured_recent))
        self._rollover_recent_token_budget = max(512, self._settings.context_rollover_recent_token_budget)
        self._rollover_summary_token_budget = 256

    def set_engine(self, engine: Engine) -> None:
        """Asigna un nuevo motor C++ LiteRT y limpia el mapa de conversaciones previas."""
        self._engine = engine
        self._conversations.clear()

    def clear(self) -> None:
        """Limpia las conversaciones activas."""
        self._conversations.clear()


    async def get_or_create(
        self,
        conversation_id: str,
        *,
        bootstrap_messages: list[dict[str, Any]],
        bootstrap_system_message: str | None = None,
        initialized_with_profile: bool = False,
        thinking_enabled: bool = False,
    ) -> ConversationState:
        async with self._manager_lock:
            state = self._conversations.get(conversation_id)
            if state is not None:
                state.touch()
                logger.info("Reusing existing conversation: %s", conversation_id)
                return state

            await self._evict_if_needed_locked()
            logger.info("Creating new conversation: %s (thinking=%s)", conversation_id, thinking_enabled)
            prepared_bootstrap_messages, bootstrap_summary, bootstrap_recent_messages = (
                self._prepare_bootstrap_context(
                    bootstrap_messages=bootstrap_messages,
                    bootstrap_system_message=bootstrap_system_message or "",
                )
            )

            conversation = await self._create_conversation(
                bootstrap_messages=prepared_bootstrap_messages,
                bootstrap_system_message=bootstrap_system_message,
                thinking_enabled=thinking_enabled,
            )
            state = ConversationState(
                conversation_id=conversation_id,
                conversation=conversation,
                bootstrap_system_message=bootstrap_system_message or "",
                rolling_messages=bootstrap_recent_messages,
                summary_text=bootstrap_summary,
                initialized_with_profile=initialized_with_profile,
            )
            state.last_known_token_count = self._estimate_context_tokens(
                bootstrap_system_message or "",
                state.summary_text,
                state.rolling_messages,
            )
            self._conversations[conversation_id] = state
            if initialized_with_profile:
                logger.info("Conversation initialized with global model profile: %s", conversation_id)
            return state

    def _prepare_bootstrap_context(
        self,
        *,
        bootstrap_messages: list[dict[str, Any]],
        bootstrap_system_message: str,
    ) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
        filtered_messages = self._filter_conversation_messages(bootstrap_messages)
        recent_messages = self._trim_recent_messages(filtered_messages)
        bootstrap_tokens = self._estimate_context_tokens(
            bootstrap_system_message,
            "",
            filtered_messages,
        )

        if bootstrap_tokens <= self._rollover_threshold_tokens:
            return bootstrap_messages, "", recent_messages

        older_messages = filtered_messages[: max(0, len(filtered_messages) - len(recent_messages))]
        summary_text = self._compact_summary_text(
            self._build_bootstrap_recovery_summary(older_messages)
        )
        compact_messages = self._build_rollover_messages(summary_text, recent_messages)

        while recent_messages and self._estimate_context_tokens(
            bootstrap_system_message,
            summary_text,
            recent_messages,
        ) > self._rollover_threshold_tokens:
            recent_messages = recent_messages[1:]
            compact_messages = self._build_rollover_messages(summary_text, recent_messages)

        after_tokens = self._estimate_context_tokens(
            bootstrap_system_message,
            summary_text,
            recent_messages,
        )
        if after_tokens > self._rollover_threshold_tokens:
            summary_text = ""
            compact_messages = list(recent_messages)
            while recent_messages and self._estimate_context_tokens(
                bootstrap_system_message,
                "",
                recent_messages,
            ) > self._rollover_threshold_tokens:
                recent_messages = recent_messages[1:]
                compact_messages = list(recent_messages)
            after_tokens = self._estimate_context_tokens(
                bootstrap_system_message,
                "",
                recent_messages,
            )

        logger.warning(
            "Compacted oversized bootstrap before conversation create: before_tokens=%s after_tokens=%s original_messages=%s recent_messages=%s",
            bootstrap_tokens,
            after_tokens,
            len(filtered_messages),
            len(recent_messages),
        )
        return compact_messages, summary_text, recent_messages

    def _build_bootstrap_recovery_summary(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            return ""

        transcript = self._messages_to_transcript(messages)
        if not transcript:
            return ""

        lines = transcript.splitlines()
        head = lines[:4]
        tail = lines[-8:] if len(lines) > 8 else lines
        summary_parts = [
            "Recovered prior chat context after inactivity.",
            "Earlier context:",
            *head,
        ]
        if tail != head:
            summary_parts.extend(["Recent older context:", *tail])
        return "\n".join(summary_parts)

    async def prepare_for_turn(
        self,
        state: ConversationState,
        incoming_payload: str | dict[str, Any],
        thinking_enabled: bool = False,
    ) -> None:
        current_tokens = self._safe_token_count(state.conversation)
        incoming_tokens = self._estimate_tokens_from_payload(incoming_payload)
        projected_tokens = current_tokens + incoming_tokens
        state.last_known_token_count = current_tokens

        # Garantizar margen suficiente para generación completa sin corte abrupto.
        # Con max_num_tokens (ej. 4096):
        # Si thinking está activo: reservamos thinking_token_budget + headroom.
        # Sin thinking: reservamos un margen holgado de generación (context_rollover_headroom_tokens, default 1500 tokens)
        # para que respuestas largas de código o explicaciones extensas NUNCA se corten a mitad de camino.
        max_tokens = self._settings.max_num_tokens
        base_headroom = getattr(self._settings, "context_rollover_headroom_tokens", 1500)
        headroom = (self._settings.thinking_token_budget + base_headroom) if thinking_enabled else base_headroom
        dynamic_threshold = min(
            self._rollover_threshold_tokens,
            max_tokens - headroom,
        )

        if projected_tokens <= dynamic_threshold:
            return

        logger.info(
            "[ROLLOVER] Activando rollover preventivo (current=%d, incoming=%d, projected=%d, threshold=%d, max=%d, thinking=%s) para %s",
            current_tokens,
            incoming_tokens,
            projected_tokens,
            dynamic_threshold,
            max_tokens,
            thinking_enabled,
            state.conversation_id,
        )
        await self._perform_context_rollover(
            state,
            current_tokens=current_tokens,
            projected_tokens=projected_tokens,
            thinking_enabled=thinking_enabled,
        )

    async def register_turn(
        self,
        state: ConversationState,
        incoming_payload: str | dict[str, Any],
        assistant_text: str,
    ) -> None:
        user_message = self._payload_to_user_message(incoming_payload)
        if user_message is not None:
            state.rolling_messages.append(user_message)
        if assistant_text.strip():
            state.rolling_messages.append({"role": "assistant", "content": assistant_text})

        # Mantener un historial amplio de turnos para resumir cuando se alcance el límite
        if len(state.rolling_messages) > 40:
            state.rolling_messages = state.rolling_messages[-40:]

        state.last_known_token_count = self._estimate_context_tokens(
            state.bootstrap_system_message,
            state.summary_text,
            state.rolling_messages,
        )
        state.touch()

    async def _perform_context_rollover(
        self,
        state: ConversationState,
        *,
        current_tokens: int,
        projected_tokens: int,
        thinking_enabled: bool = False,
    ) -> None:
        # 1. Separar mensajes recientes de mensajes antiguos
        recent_messages = self._select_recent_messages(
            state.rolling_messages,
            self._rollover_recent_messages,
            self._rollover_recent_token_budget,
        )
        recent_ids = {id(m) for m in recent_messages}
        older_messages = [m for m in state.rolling_messages if id(m) not in recent_ids]

        # 2. Generar resumen compacto de los mensajes antiguos + resumen previo
        summary_text = await self._summarize_context(state, older_messages=older_messages)
        merged_summary = summary_text.strip() or state.summary_text.strip()
        compact_summary = self._compact_summary_text(merged_summary)

        # 3. Formatear el system prompt con la memoria resumida inyectada
        base_system = state.bootstrap_system_message.strip()
        if compact_summary:
            memory_block = (
                f"\n\n[Resumen del contexto previo de la conversación]:\n{compact_summary}\n"
                "[Fin del resumen de contexto previo. Continúa respondiendo normalmente a los últimos mensajes.]"
            )
            rollover_system_prompt = f"{base_system}{memory_block}" if base_system else memory_block
        else:
            rollover_system_prompt = base_system

        # 4. CRÍTICO PARA EVITAR OOM EN LÍMITE DE 3 GB:
        # Cerrar y destruir la conversación previa en C++ y forzar garbage collection
        # ANTES de instanciar la nueva. Si se crea la nueva antes de cerrar la antigua,
        # coexisten dos KV-caches masivos en RAM superando los 3.5 GB y congelando el NAS.
        old_conversation = state.conversation
        state.conversation = None
        if hasattr(old_conversation, "close"):
            try:
                await asyncio.to_thread(old_conversation.close)
            except Exception:
                logger.exception("Error closing old conversation during rollover for %s", state.conversation_id)

        force_garbage_collection()

        # 5. Crear la nueva conversación en LiteRT con KV-cache limpio (~300-500 tokens)
        new_conversation = await self._create_conversation(
            bootstrap_messages=recent_messages,
            bootstrap_system_message=rollover_system_prompt,
            thinking_enabled=thinking_enabled,
        )

        state.conversation = new_conversation
        state.summary_text = compact_summary
        state.rolling_messages = list(recent_messages)
        state.rollover_count += 1


        post_tokens = self._estimate_context_tokens(
            rollover_system_prompt,
            "",
            state.rolling_messages,
        )
        state.last_known_token_count = post_tokens

        logger.info(
            "[ROLLOVER COMPLETADO] conversación=%s antes=%d tokens, proyectado=%d, después=%d tokens, recientes=%d, rollovers=%d",
            state.conversation_id,
            current_tokens,
            projected_tokens,
            post_tokens,
            len(recent_messages),
            state.rollover_count,
        )

    async def _summarize_context(
        self,
        state: ConversationState,
        older_messages: list[dict[str, Any]] | None = None,
    ) -> str:
        msgs_to_summarize = older_messages if older_messages is not None else state.rolling_messages
        transcript = self._messages_to_transcript(msgs_to_summarize)
        if not transcript and state.summary_text.strip():
            return state.summary_text.strip()

        if not self._settings.enable_admin_llm:
            return self._build_structured_summary(msgs_to_summarize, state.summary_text)

        summary_prompt = self._build_summary_prompt(
            previous_summary=state.summary_text,
            transcript=transcript,
        )

        summarizer_conversation = None
        try:
            summarizer_conversation = await self._create_conversation(
                bootstrap_messages=[],
                bootstrap_system_message=(
                    "You compress conversation memory for long chats. "
                    "Return only a concise factual summary in plain text."
                ),
            )
            summary_response = await asyncio.to_thread(
                summarizer_conversation.send_message,
                summary_prompt,
            )
            summary_text = sdk_message_to_text(summary_response).strip()
            if summary_text:
                return summary_text
        except Exception:
            logger.exception("Failed to generate rollover summary for %s", state.conversation_id)
        finally:
            if summarizer_conversation is not None and hasattr(summarizer_conversation, "close"):
                try:
                    await asyncio.to_thread(summarizer_conversation.close)
                except Exception:
                    pass

        return self._build_structured_summary(msgs_to_summarize, state.summary_text)

    def _build_structured_summary(
        self,
        older_messages: list[dict[str, Any]],
        previous_summary: str,
    ) -> str:
        summary_lines: list[str] = []
        if previous_summary.strip():
            summary_lines.append(f"Contexto previo acumulado: {previous_summary.strip()}")

        topics: list[str] = []
        for msg in older_messages:
            if msg.get("role") == "user":
                text = self._content_to_text(msg.get("content")).strip()
                if text:
                    first_line = text.splitlines()[0]
                    words = first_line.split()[:14]
                    topic = " ".join(words)
                    if topic and topic not in topics:
                        topics.append(topic)

        if topics:
            topics_str = " | ".join(topics[-8:])
            summary_lines.append(f"Temas tratados: {topics_str}")

        return "\n".join(summary_lines)

    def _build_summary_prompt(self, *, previous_summary: str, transcript: str) -> str:
        parts = [
            "Summarize the conversation context for memory compaction.",
            "Keep only durable facts, user preferences, decisions, unresolved tasks, and hard constraints.",
            "Rewrite the previous summary together with the recent transcript.",
            "Maximum 40 words. No markdown. No preamble.",
        ]
        if previous_summary.strip():
            parts.append("Previous summary:")
            parts.append(previous_summary.strip())
        if transcript.strip():
            parts.append("Recent transcript:")
            parts.append(transcript.strip())
        return "\n\n".join(parts)


    async def _create_conversation(
        self,
        *,
        bootstrap_messages: list[dict[str, Any]],
        bootstrap_system_message: str | None,
        thinking_enabled: bool = False,
    ) -> Conversation:
        conversation_kwargs: dict[str, Any] = {"messages": bootstrap_messages}
        try:
            create_signature = inspect.signature(self._engine.create_conversation)
            if bootstrap_system_message and "system_message" in create_signature.parameters:
                conversation_kwargs["system_message"] = bootstrap_system_message
            if "filter_channel_content_from_kv_cache" in create_signature.parameters:
                conversation_kwargs["filter_channel_content_from_kv_cache"] = True
            if "thinking_config" in create_signature.parameters:
                from litert_lm.interfaces import ThinkingConfig

                if thinking_enabled:
                    conversation_kwargs["thinking_config"] = ThinkingConfig(
                        enable_thinking=True,
                        thinking_token_budget=self._settings.thinking_token_budget,
                    )
                else:
                    conversation_kwargs["thinking_config"] = ThinkingConfig(
                        enable_thinking=False,
                    )
            if "sampler_config" in create_signature.parameters:
                try:
                    from app.profile_store import get_profile_store
                    gen_defaults = get_profile_store().profile.generation_params
                    if "temperature" in gen_defaults or "top_p" in gen_defaults:
                        from litert_lm.interfaces import SamplerConfig
                        temp = float(gen_defaults.get("temperature", 0.7))
                        top_p = float(gen_defaults.get("top_p", 0.9))
                        conversation_kwargs["sampler_config"] = SamplerConfig(
                            temperature=temp,
                            top_p=top_p,
                        )
                except Exception:
                    pass
        except (TypeError, ValueError):
            pass

        return await asyncio.to_thread(
            self._engine.create_conversation,
            **conversation_kwargs,
        )

    def _build_rollover_messages(
        self,
        summary_text: str,
        recent_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if summary_text.strip():
            messages.append({"role": "developer", "content": summary_text.strip()})
        messages.extend(recent_messages)
        return messages

    def _select_recent_messages(
        self,
        messages: list[dict[str, Any]],
        max_pairs: int,
        token_budget: int,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []

        roles = {"user", "assistant"}
        filtered = [m for m in messages if m.get("role") in roles]
        if not filtered:
            return []

        selected: list[dict[str, Any]] = []
        assistant_count = 0
        estimated_tokens = 0
        for message in reversed(filtered):
            message_tokens = self._estimate_message_tokens(message)
            if selected and (len(selected) >= max_pairs * 2 or estimated_tokens + message_tokens > token_budget):
                break
            selected.append(message)
            estimated_tokens += message_tokens
            if message.get("role") == "assistant":
                assistant_count += 1
                if assistant_count >= max_pairs:
                    break
        selected.reverse()
        return selected

    def _trim_recent_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not messages:
            return []

        return self._select_recent_messages(
            messages,
            self._rollover_recent_messages,
            self._rollover_recent_token_budget,
        )

    def _estimate_context_tokens(
        self,
        system_prompt: str,
        summary_text: str,
        messages: list[dict[str, Any]],
    ) -> int:
        pieces = [system_prompt.strip(), summary_text.strip(), self._messages_to_transcript(messages)]
        combined = "\n\n".join(piece for piece in pieces if piece)
        if not combined:
            return 0

        try:
            token_ids = self._engine.tokenize(combined)
            if isinstance(token_ids, list):
                return len(token_ids)
        except Exception:
            logger.exception("Failed context token estimate")
        return 0

    def _compact_summary_text(self, summary_text: str) -> str:
        text = summary_text.strip()
        if not text:
            return ""

        try:
            token_ids = self._engine.tokenize(text)
            if isinstance(token_ids, list) and len(token_ids) > self._rollover_summary_token_budget:
                token_ids = token_ids[: self._rollover_summary_token_budget]
                compacted = self._engine.detokenize(token_ids).strip()
                return compacted or text[:1024].strip()
        except Exception:
            logger.exception("Failed to compact rollover summary")
        return text[:1024].strip()

    def _messages_to_transcript(self, messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in messages:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = self._content_to_text(message.get("content"))
            if not content:
                continue
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, dict):
            content_type = content.get("type")
            if content_type in {"text", "input_text"} and isinstance(content.get("text"), str):
                return content["text"].strip()
            return normalize_text_content(content).strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                    continue

                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type in {"text", "input_text"} and isinstance(item.get("text"), str):
                    parts.append(item["text"].strip())
                elif item_type in {"image", "image_url"}:
                    parts.append("[image]")
                elif item_type in {"audio", "input_audio"}:
                    parts.append("[audio]")
            return "\n".join(part for part in parts if part)

        return ""

    def _estimate_tokens_from_payload(self, incoming_payload: str | dict[str, Any]) -> int:
        incoming_text = self._payload_to_text(incoming_payload)
        if not incoming_text:
            return 0

        try:
            token_ids = self._engine.tokenize(incoming_text)
            if isinstance(token_ids, list):
                return len(token_ids)
        except Exception:
            logger.exception("Failed token estimate during rollover projection")
        return 0

    def _estimate_message_tokens(self, message: dict[str, Any]) -> int:
        content = self._content_to_text(message.get("content"))
        if not content:
            return 0

        try:
            token_ids = self._engine.tokenize(content)
            if isinstance(token_ids, list):
                return len(token_ids)
        except Exception:
            logger.exception("Failed message token estimate")
        return max(1, len(content) // 4)

    def _payload_to_text(self, incoming_payload: str | dict[str, Any]) -> str:
        if isinstance(incoming_payload, str):
            return incoming_payload.strip()

        if isinstance(incoming_payload, dict):
            content = incoming_payload.get("content", "")
            return self._content_to_text(content)

        return ""

    def _payload_to_user_message(self, incoming_payload: str | dict[str, Any]) -> dict[str, Any] | None:
        if isinstance(incoming_payload, str):
            text = incoming_payload.strip()
            if not text:
                return None
            return {"role": "user", "content": text}

        if not isinstance(incoming_payload, dict):
            return None

        role = incoming_payload.get("role", "user")
        if role != "user":
            role = "user"
        content = incoming_payload.get("content")
        if not self._content_to_text(content):
            return None
        return {"role": role, "content": content}

    def _filter_conversation_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            filtered.append({"role": role, "content": message.get("content", "")})
        return filtered

    def _safe_token_count(self, conversation: Conversation) -> int:
        try:
            return int(conversation.token_count)
        except Exception:
            return 0

    async def _evict_if_needed_locked(self) -> None:
        max_active = self._settings.max_active_conversations
        if len(self._conversations) < max_active:
            return

        removable = sorted(
            self._conversations.values(),
            key=lambda item: item.last_access,
        )

        for candidate in removable:
            if candidate.lock.locked():
                continue
            await self._delete_locked(candidate.conversation_id)
            logger.warning("Evicted conversation due to max limit: %s", candidate.conversation_id)
            break

    async def cleanup_expired(self) -> int:
        timeout = self._settings.session_timeout
        now = now_ts()

        async with self._manager_lock:
            expired_ids = [
                conv_id
                for conv_id, state in self._conversations.items()
                if (now - state.last_access) > timeout and not state.lock.locked()
            ]

            for conv_id in expired_ids:
                await self._delete_locked(conv_id)

        if expired_ids:
            logger.info("Cleaned up %s expired conversations", len(expired_ids))
        return len(expired_ids)

    async def _delete_locked(self, conversation_id: str) -> None:
        state = self._conversations.pop(conversation_id, None)
        if state is None:
            return
        
        # Evitar excepciones si el objeto Conversation del SDK no expone .close()
        if hasattr(state.conversation, "close"):
            try:
                await asyncio.to_thread(state.conversation.close)
            except Exception:
                logger.exception("Error closing conversation backend thread for %s", conversation_id)

        force_garbage_collection()

    async def close_all(self) -> None:
        async with self._manager_lock:
            all_ids = list(self._conversations.keys())
            for conversation_id in all_ids:
                await self._delete_locked(conversation_id)

    async def stats(self) -> dict[str, int]:
        async with self._manager_lock:
            active_count = len(self._conversations)
            initialized_with_profile_count = sum(
                1
                for state in self._conversations.values()
                if state.initialized_with_profile
            )
        return {
            "active_conversations": active_count,
            "profile_initialized_conversations": initialized_with_profile_count,
        }


_conversation_manager: ConversationManager | None = None
_manager_lock = asyncio.Lock()


async def init_conversation_manager(engine: Engine | None = None) -> ConversationManager:
    global _conversation_manager

    if _conversation_manager is not None:
        if engine is not None:
            _conversation_manager.set_engine(engine)
        return _conversation_manager

    async with _manager_lock:
        if _conversation_manager is None:
            _conversation_manager = ConversationManager(engine)
        elif engine is not None:
            _conversation_manager.set_engine(engine)
        return _conversation_manager


def get_conversation_manager() -> ConversationManager:
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager(None)
    return _conversation_manager


async def close_conversation_manager() -> None:
    global _conversation_manager

    async with _manager_lock:
        if _conversation_manager is None:
            return
        await _conversation_manager.close_all()
        _conversation_manager = None

