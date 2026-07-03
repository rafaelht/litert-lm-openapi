from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

from litert_lm import Conversation, Engine

from app.config import get_settings
from app.utils import now_ts

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    conversation_id: str
    conversation: Conversation
    initialized_with_profile: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_access: float = field(default_factory=now_ts)

    def touch(self) -> None:
        self.last_access = now_ts()


class ConversationManager:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._settings = get_settings()
        self._conversations: dict[str, ConversationState] = {}
        self._manager_lock = asyncio.Lock()

    async def get_or_create(
        self,
        conversation_id: str,
        *,
        bootstrap_messages: list[dict[str, Any]],
        bootstrap_system_message: str | None = None,
        initialized_with_profile: bool = False,
    ) -> ConversationState:
        async with self._manager_lock:
            state = self._conversations.get(conversation_id)
            if state is not None:
                state.touch()
                logger.info("Reusing existing conversation: %s", conversation_id)
                return state

            await self._evict_if_needed_locked()
            logger.info("Creating new conversation: %s", conversation_id)
            conversation_kwargs: dict[str, Any] = {"messages": bootstrap_messages}
            if bootstrap_system_message:
                try:
                    create_signature = inspect.signature(self._engine.create_conversation)
                    if "system_message" in create_signature.parameters:
                        conversation_kwargs["system_message"] = bootstrap_system_message
                except (TypeError, ValueError):
                    pass

            conversation = await asyncio.to_thread(
                self._engine.create_conversation,
                **conversation_kwargs,
            )
            state = ConversationState(
                conversation_id=conversation_id,
                conversation=conversation,
                initialized_with_profile=initialized_with_profile,
            )
            self._conversations[conversation_id] = state
            if initialized_with_profile:
                logger.info("Conversation initialized with global model profile: %s", conversation_id)
            return state

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


async def init_conversation_manager(engine: Engine) -> ConversationManager:
    global _conversation_manager

    if _conversation_manager is not None:
        return _conversation_manager

    async with _manager_lock:
        if _conversation_manager is None:
            _conversation_manager = ConversationManager(engine)
        return _conversation_manager


def get_conversation_manager() -> ConversationManager:
    if _conversation_manager is None:
        raise RuntimeError("Conversation manager is not initialized")
    return _conversation_manager


async def close_conversation_manager() -> None:
    global _conversation_manager

    async with _manager_lock:
        if _conversation_manager is None:
            return
        await _conversation_manager.close_all()
        _conversation_manager = None