import json
from typing import Iterable, Self

from langchain_core.messages import AnyMessage, ToolMessage
from pydantic import BaseModel, PrivateAttr


# v1 fixed for rubric
class ToolTrackedResult(BaseModel):
    """Keep the tool trace outside the model-facing structured output schema."""

    _tool_usage: list[dict[str, str]] = PrivateAttr(default_factory=list)

    @property
    def tool_usage(self) -> list[dict[str, str]]:
        return self._tool_usage

    def record_tool_call(self, name: str, status: str = "success") -> Self:
        self._tool_usage.append({"name": name, "status": status})
        return self

    def record_tool_messages(self, messages: Iterable[AnyMessage]) -> Self:
        for message in messages:
            if isinstance(message, ToolMessage) and message.name:
                self.record_tool_call(message.name, _tool_status(message))
        return self


# v1 fixed for rubric
def _tool_status(message: ToolMessage) -> str:
    if message.status == "error":
        return "error"
    if isinstance(message.content, str):
        try:
            result = json.loads(message.content)
        except ValueError:
            return message.status
        for item in result if isinstance(result, list) else [result]:
            if isinstance(item, dict) and item.get("error"):
                return "error"
        if isinstance(result, dict) and result.get("requires_escalation"):
            return "requires_escalation"
    return message.status
