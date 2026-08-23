from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

from litert_lm import Conversation, Engine, Tool

from app.config import get_settings
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
    tools: list[Tool] = field(default_factory=list)
    tool_signature: str = ""
    automatic_tool_calling: bool = True
    extra_context: dict[str, Any] = field(default_factory=dict)
    extra_context_signature: str = ""
    filter_channel_content_from_kv_cache: bool = False
    thinking_config: Any = None
    sampler_config: Any = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_access: float = field(default_factory=now_ts)

    def touch(self) -> None:
        self.last_access = now_ts()


class ConversationManager:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._settings = get_settings()
        self._conversations: dict[str, ConversationState] = {}
        self._warm_pool: dict[str, Any] = {}
        self._warming_in_progress: set[str] = set()
        self._manager_lock = asyncio.Lock()
        self._rollover_threshold_tokens = max(1, self._settings.context_rollover_threshold_tokens)
        configured_recent = self._settings.context_rollover_recent_messages
        self._rollover_recent_messages = min(3, max(1, configured_recent))
        self._rollover_recent_token_budget = max(64, self._settings.context_rollover_recent_token_budget)
        self._rollover_summary_token_budget = 64
        if configured_recent != self._rollover_recent_messages:
            logger.warning(
                "CONTEXT_ROLLOVER_RECENT_MESSAGES=%s is out of supported range [1,3]; using %s",
                configured_recent,
                self._rollover_recent_messages,
            )

    async def get_or_create(
        self,
        conversation_id: str,
        *,
        bootstrap_messages: list[dict[str, Any]],
        bootstrap_system_message: str | None = None,
        initialized_with_profile: bool = False,
        tools: list[Tool] | None = None,
        tool_signature: str = "",
        automatic_tool_calling: bool = True,
        extra_context: dict[str, Any] | None = None,
        extra_context_signature: str = "",
        filter_channel_content_from_kv_cache: bool = False,
        thinking_config: Any = None,
        sampler_config: Any = None,
    ) -> ConversationState:
        async with self._manager_lock:
            state = self._conversations.get(conversation_id)
            if state is not None:
                if (
                    state.tool_signature != tool_signature
                    or state.extra_context_signature != extra_context_signature
                ):
                    await self._refresh_conversation_options_locked(
                        state,
                        tools=tools,
                        tool_signature=tool_signature,
                        automatic_tool_calling=automatic_tool_calling,
                        extra_context=extra_context,
                        extra_context_signature=extra_context_signature,
                        filter_channel_content_from_kv_cache=filter_channel_content_from_kv_cache,
                    )
                state.touch()
                logger.info("Reusing existing conversation: %s", conversation_id)
                return state

            await self._evict_if_needed_locked()
            logger.info("Creating new conversation: %s", conversation_id)
            prepared_bootstrap_messages, bootstrap_summary, bootstrap_recent_messages = (
                self._prepare_bootstrap_context(
                    bootstrap_messages=bootstrap_messages,
                    bootstrap_system_message=bootstrap_system_message or "",
                )
            )

            warm_conv = None
            if bootstrap_system_message:
                pool_key = hashlib.sha256(bootstrap_system_message.encode()).hexdigest()[:16]
                warm_conv = self._warm_pool.pop(pool_key, None)

            conversation_kwargs = self._conversation_kwargs(
                bootstrap_messages=prepared_bootstrap_messages,
                bootstrap_system_message=bootstrap_system_message,
                tools=tools,
                automatic_tool_calling=automatic_tool_calling,
                extra_context=extra_context,
                filter_channel_content_from_kv_cache=filter_channel_content_from_kv_cache,
                thinking_config=thinking_config,
                sampler_config=sampler_config,
            )

            if warm_conv is not None:
                logger.info("Using pre-warmed conversation for %s (TTFT saved)", conversation_id)
                conversation = warm_conv
                if prepared_bootstrap_messages:
                    for msg in prepared_bootstrap_messages:
                        try:
                            # send_message processes the tokens for the history message
                            # Wait, the SDK needs to know if this is a user or assistant msg.
                            # send_message normally expects the payload for generation.
                            # Actually, if we use a pre-warmed conversation, we can't easily inject a list of messages.
                            # So the warm pool is ONLY useful if bootstrap_messages is empty.
                            # Otherwise, we fallback to create_conversation.
                            pass
                        except Exception:
                            pass
                    # Let's fix this in the logic.
            
            # Re-evaluate warm_conv logic: we can only use it if prepared_bootstrap_messages is empty
            if warm_conv is not None and not prepared_bootstrap_messages and not tools:
                logger.info("Using pre-warmed conversation for %s (TTFT saved)", conversation_id)
                conversation = warm_conv
                # Start warming the next one in background
                asyncio.create_task(self.warm_system_prompt(bootstrap_system_message))
            else:
                if warm_conv is not None:
                    # Put it back since we couldn't use it
                    pool_key = hashlib.sha256(bootstrap_system_message.encode()).hexdigest()[:16]
                    self._warm_pool[pool_key] = warm_conv
                    
                conversation = await asyncio.to_thread(
                    self._engine.create_conversation,
                    **conversation_kwargs,
                )

            effective_tools = tools or []
            if tools:
                conversation, effective_tools = await self._drop_tools_if_context_is_too_large(
                    conversation,
                    conversation_kwargs,
                    conversation_id=conversation_id,
                )
                
            state = ConversationState(
                conversation_id=conversation_id,
                conversation=conversation,
                bootstrap_system_message=bootstrap_system_message or "",
                rolling_messages=bootstrap_recent_messages,
                summary_text=bootstrap_summary,
                initialized_with_profile=initialized_with_profile,
                tools=effective_tools,
                tool_signature=tool_signature,
                automatic_tool_calling=automatic_tool_calling,
                extra_context=extra_context or {},
                extra_context_signature=extra_context_signature,
                filter_channel_content_from_kv_cache=filter_channel_content_from_kv_cache,
                thinking_config=thinking_config,
                sampler_config=sampler_config,
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

    async def _refresh_conversation_options_locked(
        self,
        state: ConversationState,
        *,
        tools: list[Tool] | None,
        tool_signature: str,
        automatic_tool_calling: bool,
        extra_context: dict[str, Any] | None,
        extra_context_signature: str,
        filter_channel_content_from_kv_cache: bool,
    ) -> None:
        bootstrap_messages = self._build_rollover_messages(
            state.summary_text,
            state.rolling_messages,
        )
        new_conversation = await self._create_conversation(
            bootstrap_messages=bootstrap_messages,
            bootstrap_system_message=state.bootstrap_system_message,
            tools=tools,
            automatic_tool_calling=automatic_tool_calling,
            extra_context=extra_context,
            filter_channel_content_from_kv_cache=filter_channel_content_from_kv_cache,
            thinking_config=state.thinking_config,
            sampler_config=state.sampler_config,
        )
        effective_tools = tools or []
        if tools:
            new_conversation, effective_tools = await self._drop_tools_if_context_is_too_large(
                new_conversation,
                self._conversation_kwargs(
                    bootstrap_messages=bootstrap_messages,
                    bootstrap_system_message=state.bootstrap_system_message,
                    tools=tools,
                    automatic_tool_calling=automatic_tool_calling,
                    extra_context=extra_context,
                    filter_channel_content_from_kv_cache=filter_channel_content_from_kv_cache,
                    thinking_config=state.thinking_config,
                    sampler_config=state.sampler_config,
                ),
                conversation_id=state.conversation_id,
            )

        old_conversation = state.conversation
        state.conversation = new_conversation
        state.tools = effective_tools
        state.tool_signature = tool_signature
        state.automatic_tool_calling = automatic_tool_calling
        state.extra_context = extra_context or {}
        state.extra_context_signature = extra_context_signature
        state.filter_channel_content_from_kv_cache = filter_channel_content_from_kv_cache
        if hasattr(old_conversation, "close"):
            try:
                await asyncio.to_thread(old_conversation.close)
            except Exception:
                logger.exception("Error closing old conversation while refreshing options for %s", state.conversation_id)

        logger.info("Refreshed conversation options for %s", state.conversation_id)

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
    ) -> None:
        current_tokens = self._safe_token_count(state.conversation)
        incoming_tokens = self._estimate_tokens_from_payload(incoming_payload)
        projected_tokens = current_tokens + incoming_tokens
        state.last_known_token_count = current_tokens

        if projected_tokens <= self._rollover_threshold_tokens:
            return

        await self._perform_context_rollover(
            state,
            current_tokens=current_tokens,
            projected_tokens=projected_tokens,
        )
        await self._ensure_turn_fits_after_rollover(
            state,
            incoming_tokens=incoming_tokens,
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
        state.rolling_messages = self._trim_recent_messages(state.rolling_messages)
        state.last_known_token_count = self._estimate_context_tokens(
            state.bootstrap_system_message,
            state.summary_text,
            state.rolling_messages,
        )
        state.touch()

    async def recover_from_context_overflow(self, state: ConversationState) -> None:
        logger.warning(
            "Recovering conversation=%s from SDK context overflow with minimal context",
            state.conversation_id,
        )
        summary_text = self._compact_summary_text(state.summary_text or self._fallback_summary(state))
        bootstrap_messages = self._build_rollover_messages(summary_text, [])
        new_conversation = await self._create_conversation(
            bootstrap_messages=bootstrap_messages,
            bootstrap_system_message=state.bootstrap_system_message,
            tools=[],
            automatic_tool_calling=state.automatic_tool_calling,
            extra_context={},
            filter_channel_content_from_kv_cache=state.filter_channel_content_from_kv_cache,
            thinking_config=state.thinking_config,
            sampler_config=state.sampler_config,
        )

        actual_tokens = self._safe_token_count(new_conversation)
        if actual_tokens > self._rollover_threshold_tokens:
            if hasattr(new_conversation, "close"):
                try:
                    await asyncio.to_thread(new_conversation.close)
                except Exception:
                    logger.exception("Error closing oversized recovery conversation for %s", state.conversation_id)
            logger.warning(
                "Recovery context still too large with system prompt for conversation=%s tokens=%s; retrying without system prompt",
                state.conversation_id,
                actual_tokens,
            )
            new_conversation = await self._create_conversation(
                bootstrap_messages=bootstrap_messages,
                bootstrap_system_message=None,
                tools=[],
                automatic_tool_calling=state.automatic_tool_calling,
                extra_context={},
                filter_channel_content_from_kv_cache=state.filter_channel_content_from_kv_cache,
                thinking_config=state.thinking_config,
                sampler_config=state.sampler_config,
            )
            summary_text = ""

        old_conversation = state.conversation
        state.conversation = new_conversation
        state.summary_text = summary_text
        state.rolling_messages = []
        state.tools = []
        state.extra_context = {}
        state.last_known_token_count = self._safe_token_count(new_conversation)
        state.rollover_count += 1
        if hasattr(old_conversation, "close"):
            try:
                await asyncio.to_thread(old_conversation.close)
            except Exception:
                logger.exception("Error closing old conversation during overflow recovery for %s", state.conversation_id)

    async def _perform_context_rollover(
        self,
        state: ConversationState,
        *,
        current_tokens: int,
        projected_tokens: int,
    ) -> None:
        summary_text = await self._summarize_context(state)
        recent_messages = self._select_recent_messages(
            state.rolling_messages,
            self._rollover_recent_messages,
            self._rollover_recent_token_budget,
        )

        merged_summary = summary_text.strip() or state.summary_text.strip()
        compact_summary = self._compact_summary_text(merged_summary)
        rollover_messages = self._build_rollover_messages(compact_summary, recent_messages)

        new_conversation = await self._create_conversation(
            bootstrap_messages=rollover_messages,
            bootstrap_system_message=state.bootstrap_system_message,
            tools=state.tools,
            automatic_tool_calling=state.automatic_tool_calling,
            extra_context=state.extra_context,
            filter_channel_content_from_kv_cache=state.filter_channel_content_from_kv_cache,
            thinking_config=state.thinking_config,
            sampler_config=state.sampler_config,
        )
        if state.tools:
            new_conversation, effective_tools = await self._drop_tools_if_context_is_too_large(
                new_conversation,
                self._conversation_kwargs(
                    bootstrap_messages=rollover_messages,
                    bootstrap_system_message=state.bootstrap_system_message,
                    tools=state.tools,
                    automatic_tool_calling=state.automatic_tool_calling,
                    extra_context=state.extra_context,
                    filter_channel_content_from_kv_cache=state.filter_channel_content_from_kv_cache,
                    thinking_config=state.thinking_config,
                    sampler_config=state.sampler_config,
                ),
                conversation_id=state.conversation_id,
            )
            state.tools = effective_tools

        old_conversation = state.conversation
        state.conversation = new_conversation
        state.summary_text = compact_summary
        state.rolling_messages = list(recent_messages)
        state.rollover_count += 1

        if hasattr(old_conversation, "close"):
            try:
                await asyncio.to_thread(old_conversation.close)
            except Exception:
                logger.exception("Error closing old conversation during rollover for %s", state.conversation_id)

        post_tokens = self._estimate_context_tokens(
            state.bootstrap_system_message,
            state.summary_text,
            state.rolling_messages,
        )
        actual_post_tokens = self._safe_token_count(state.conversation)
        state.last_known_token_count = actual_post_tokens or post_tokens

        logger.warning(
            "Context rollover conversation=%s before_tokens=%s projected_tokens=%s estimated_after_tokens=%s actual_after_tokens=%s recent_messages=%s rollovers=%s",
            state.conversation_id,
            current_tokens,
            projected_tokens,
            post_tokens,
            actual_post_tokens,
            len(recent_messages),
            state.rollover_count,
        )

    async def _ensure_turn_fits_after_rollover(
        self,
        state: ConversationState,
        *,
        incoming_tokens: int,
    ) -> None:
        actual_tokens = self._safe_token_count(state.conversation)
        if actual_tokens + incoming_tokens <= self._rollover_threshold_tokens:
            return

        if state.tools:
            logger.warning(
                "Dropping tool schemas after rollover because actual SDK tokens still exceed budget: conversation=%s tokens=%s incoming=%s threshold=%s",
                state.conversation_id,
                actual_tokens,
                incoming_tokens,
                self._rollover_threshold_tokens,
            )
            await self._recreate_current_context(
                state,
                tools=[],
                extra_context=state.extra_context,
            )
            actual_tokens = self._safe_token_count(state.conversation)
            if actual_tokens + incoming_tokens <= self._rollover_threshold_tokens:
                return

        if state.extra_context:
            logger.warning(
                "Dropping extra context after rollover because actual SDK tokens still exceed budget: conversation=%s tokens=%s incoming=%s threshold=%s",
                state.conversation_id,
                actual_tokens,
                incoming_tokens,
                self._rollover_threshold_tokens,
            )
            await self._recreate_current_context(
                state,
                tools=state.tools,
                extra_context={},
            )

    async def _recreate_current_context(
        self,
        state: ConversationState,
        *,
        tools: list[Tool],
        extra_context: dict[str, Any],
    ) -> None:
        bootstrap_messages = self._build_rollover_messages(
            state.summary_text,
            state.rolling_messages,
        )
        new_conversation = await self._create_conversation(
            bootstrap_messages=bootstrap_messages,
            bootstrap_system_message=state.bootstrap_system_message,
            tools=tools,
            automatic_tool_calling=state.automatic_tool_calling,
            extra_context=extra_context,
            filter_channel_content_from_kv_cache=state.filter_channel_content_from_kv_cache,
            thinking_config=state.thinking_config,
            sampler_config=state.sampler_config,
        )

        old_conversation = state.conversation
        state.conversation = new_conversation
        state.tools = tools
        state.extra_context = extra_context
        state.last_known_token_count = self._safe_token_count(new_conversation)
        if hasattr(old_conversation, "close"):
            try:
                await asyncio.to_thread(old_conversation.close)
            except Exception:
                logger.exception("Error closing old conversation during budget recovery for %s", state.conversation_id)

    async def _summarize_context(self, state: ConversationState) -> str:
        transcript = self._messages_to_transcript(self._trim_recent_messages(state.rolling_messages))
        if not transcript and state.summary_text.strip():
            return state.summary_text.strip()

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
                    logger.exception("Error closing summarizer conversation for %s", state.conversation_id)

        fallback = self._fallback_summary(state)
        if fallback:
            logger.warning("Using fallback summary during rollover for %s", state.conversation_id)
        return fallback

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

    def _fallback_summary(self, state: ConversationState) -> str:
        previous_summary = state.summary_text.strip()
        transcript = self._messages_to_transcript(state.rolling_messages)
        if not transcript:
            return previous_summary

        transcript_lines = transcript.splitlines()
        tail = "\n".join(transcript_lines[-12:])
        if previous_summary:
            return f"{previous_summary}\n\nRecent context:\n{tail}".strip()
        return f"Recent context:\n{tail}".strip()

    async def _create_conversation(
        self,
        *,
        bootstrap_messages: list[dict[str, Any]],
        bootstrap_system_message: str | None,
        tools: list[Tool] | None = None,
        automatic_tool_calling: bool = True,
        extra_context: dict[str, Any] | None = None,
        filter_channel_content_from_kv_cache: bool = False,
        thinking_config: Any = None,
        sampler_config: Any = None,
    ) -> Conversation:
        conversation_kwargs = self._conversation_kwargs(
            bootstrap_messages=bootstrap_messages,
            bootstrap_system_message=bootstrap_system_message,
            tools=tools,
            automatic_tool_calling=automatic_tool_calling,
            extra_context=extra_context,
            filter_channel_content_from_kv_cache=filter_channel_content_from_kv_cache,
            thinking_config=thinking_config,
            sampler_config=sampler_config,
        )
        return await asyncio.to_thread(
            self._engine.create_conversation,
            **conversation_kwargs,
        )

    def _conversation_kwargs(
        self,
        *,
        bootstrap_messages: list[dict[str, Any]],
        bootstrap_system_message: str | None,
        tools: list[Tool] | None = None,
        automatic_tool_calling: bool = True,
        extra_context: dict[str, Any] | None = None,
        filter_channel_content_from_kv_cache: bool = False,
        thinking_config: Any = None,
        sampler_config: Any = None,
    ) -> dict[str, Any]:
        conversation_kwargs: dict[str, Any] = {"messages": bootstrap_messages}
        if tools:
            conversation_kwargs["tools"] = tools
            conversation_kwargs["automatic_tool_calling"] = automatic_tool_calling
        if extra_context:
            conversation_kwargs["extra_context"] = extra_context
        if filter_channel_content_from_kv_cache:
            conversation_kwargs["filter_channel_content_from_kv_cache"] = True
            
        try:
            create_signature = inspect.signature(self._engine.create_conversation)
            if bootstrap_system_message and "system_message" in create_signature.parameters:
                conversation_kwargs["system_message"] = bootstrap_system_message
            if thinking_config is not None and "thinking_config" in create_signature.parameters:
                conversation_kwargs["thinking_config"] = thinking_config
            if sampler_config is not None and "sampler_config" in create_signature.parameters:
                conversation_kwargs["sampler_config"] = sampler_config
        except (TypeError, ValueError):
            pass

        return conversation_kwargs

    async def _drop_tools_if_context_is_too_large(
        self,
        conversation: Conversation,
        conversation_kwargs: dict[str, Any],
        *,
        conversation_id: str,
    ) -> tuple[Conversation, list[Tool]]:
        actual_tokens = self._safe_token_count(conversation)
        if actual_tokens <= self._rollover_threshold_tokens:
            return conversation, conversation_kwargs.get("tools", [])

        tool_count = len(conversation_kwargs.get("tools", []))
        logger.warning(
            "Disabling tool schemas for conversation=%s because SDK context is already too large after create: tokens=%s threshold=%s tools=%s",
            conversation_id,
            actual_tokens,
            self._rollover_threshold_tokens,
            tool_count,
        )

        if hasattr(conversation, "close"):
            try:
                await asyncio.to_thread(conversation.close)
            except Exception:
                logger.exception("Error closing oversized tool conversation for %s", conversation_id)

        fallback_kwargs = dict(conversation_kwargs)
        fallback_kwargs.pop("tools", None)
        fallback_kwargs.pop("automatic_tool_calling", None)
        fallback_conversation = await asyncio.to_thread(
            self._engine.create_conversation,
            **fallback_kwargs,
        )
        fallback_tokens = self._safe_token_count(fallback_conversation)
        if fallback_tokens > self._rollover_threshold_tokens and fallback_kwargs.get("extra_context"):
            logger.warning(
                "Disabling extra context for conversation=%s because SDK context is still too large after dropping tools: tokens=%s threshold=%s",
                conversation_id,
                fallback_tokens,
                self._rollover_threshold_tokens,
            )
            if hasattr(fallback_conversation, "close"):
                try:
                    await asyncio.to_thread(fallback_conversation.close)
                except Exception:
                    logger.exception("Error closing oversized extra-context conversation for %s", conversation_id)
            fallback_kwargs.pop("extra_context", None)
            fallback_conversation = await asyncio.to_thread(
                self._engine.create_conversation,
                **fallback_kwargs,
            )
        return fallback_conversation, []

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

        roles = {"user", "assistant", "tool"}
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
            if role not in {"user", "assistant", "tool"}:
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
                elif item_type == "tool_response":
                    response = item.get("response")
                    if isinstance(response, str):
                        parts.append(response.strip())
                    elif response is not None:
                        parts.append(str(response))
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
        if role not in {"user", "tool"}:
            role = "user"
        content = incoming_payload.get("content")
        if not self._content_to_text(content):
            return None
        return {"role": role, "content": content}

    def _filter_conversation_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role not in {"user", "assistant", "tool"}:
                continue
            filtered_message = {"role": role, "content": message.get("content", "")}
            if "tool_calls" in message:
                filtered_message["tool_calls"] = message["tool_calls"]
            filtered.append(filtered_message)
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

    async def warm_system_prompt(self, system_message: str) -> None:
        """Pre-calienta una conversación con el system prompt dado."""
        if not system_message:
            return
        
        pool_key = hashlib.sha256(system_message.encode()).hexdigest()[:16]
        if pool_key in self._warm_pool or pool_key in self._warming_in_progress:
            return
        
        self._warming_in_progress.add(pool_key)
        try:
            warm_conv = await asyncio.to_thread(
                self._engine.create_conversation,
                system_message=system_message,
                messages=[],
            )
            self._warm_pool[pool_key] = warm_conv
            logger.info("Pre-warmed conversation for system prompt (hash=%s)", pool_key)
        except Exception:
            logger.exception("Failed to pre-warm conversation")
        finally:
            self._warming_in_progress.discard(pool_key)

    async def close_all(self) -> None:
        async with self._manager_lock:
            all_ids = list(self._conversations.keys())
            for conversation_id in all_ids:
                await self._delete_locked(conversation_id)
            for conv in self._warm_pool.values():
                if hasattr(conv, "close"):
                    try:
                        await asyncio.to_thread(conv.close)
                    except Exception:
                        logger.exception("Error closing warm pool conversation")
            self._warm_pool.clear()

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
