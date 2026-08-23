from __future__ import annotations

import hashlib
import json
from typing import Any

from litert_lm import Tool


class OpenAIToolDescriptor(Tool):
    def __init__(self, description: dict[str, Any]) -> None:
        self._description = description

    def get_tool_description(self) -> dict[str, Any]:
        return self._description

    def execute(self, param: dict[str, Any]) -> Any:
        return {
            "error": (
                "Tool execution is delegated to the OpenAI-compatible client. "
                "Run this conversation with automatic tool calling disabled."
            ),
            "arguments": param,
        }


def normalize_openai_tools(raw_tools: Any) -> list[OpenAIToolDescriptor]:
    if not isinstance(raw_tools, list):
        return []

    tools: list[OpenAIToolDescriptor] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        description = {
            "type": tool.get("type", "function"),
            "function": function,
        }
        tools.append(OpenAIToolDescriptor(description))
    return tools


def tools_signature(raw_tools: Any) -> str:
    if not raw_tools:
        return ""
    try:
        payload = json.dumps(raw_tools, ensure_ascii=False, sort_keys=True)
    except TypeError:
        payload = str(raw_tools)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
