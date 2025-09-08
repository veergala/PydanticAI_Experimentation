"""Utility functions for the chat app."""

from datetime import datetime, timezone

from models import ChatMessage
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)


def to_chat_message(m: ModelMessage) -> ChatMessage:
    """Convert a ModelMessage to a ChatMessage for the frontend."""
    if isinstance(m, ModelRequest):
        # Look for the UserPromptPart, skipping SystemPromptParts
        for part in m.parts:
            if isinstance(part, UserPromptPart):
                assert isinstance(part.content, str)
                return {
                    "role": "user",
                    "timestamp": part.timestamp.isoformat(),
                    "content": part.content,
                }
        # If no UserPromptPart found, skip this message (it's system-only)
        raise UnexpectedModelBehavior("No user prompt found in ModelRequest")

    elif isinstance(m, ModelResponse):
        first_part = m.parts[0]
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
