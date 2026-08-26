from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAIModel(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "litert-session-server"


class OpenAIModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIModel]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    stream: bool = False

    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    reasoning_effort: str | None = None
    thinking: bool | dict[str, Any] | None = None


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"] = "stop"


class ChatCompletionUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    total_duration: int | None = None
    load_duration: int | None = None


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    total_duration: int | None = None
    load_duration: int | None = None
