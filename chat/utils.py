"""Utility functions for the chat app."""

from datetime import datetime, timezone

from models import ChatMessage
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)


def to_chat_message(m: ModelMessage) -> ChatMessage:
    """Convert a ModelMessage to a ChatMessage for the frontend."""
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return {
                "role": "user",
                "timestamp": first_part.timestamp.isoformat(),
                "content": first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                "role": "model",
                "timestamp": m.timestamp.isoformat(),
                "content": first_part.content,
            }
    raise UnexpectedModelBehavior(f"Unexpected message type for chat app: {m}")


def create_user_message(content: str) -> ChatMessage:
    """Create a user ChatMessage with current timestamp."""
    return {
        "role": "user",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "content": content,
    }
