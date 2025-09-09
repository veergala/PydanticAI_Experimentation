"""Simple chat app example built with FastAPI.

Run with:
    uv run app.py
"""

from __future__ import annotations as _annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import fastapi
import logfire
from database import Database
from dotenv import load_dotenv
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from tools import register_tools
from utils import create_user_message, to_chat_message

load_dotenv()

# Configure logging
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()

# Initialize the AI agent with enhanced capabilities
agent = Agent(
    "openai:gpt-4o",
    system_prompt="""
    You are a helpful AI assistant with access to real-world weather and time data.
    Unlike basic chatbots, you can:
    - Check current weather conditions for any location
    - Get the current date and time
    - Remember context between conversations
    
    Use your tools when appropriate to provide accurate, up-to-date information.
    Be conversational and helpful!
    """,
)

# Register all the enhanced tools
register_tools(agent)

THIS_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    """Manage database connection lifecycle."""
    async with Database.connect() as db:
        yield {"db": db}


app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)


async def get_db(request: Request) -> Database:
    """Dependency to get database connection."""
    return request.state.db


@app.get("/")
async def index() -> FileResponse:
    """Serve the main chat interface."""
    return FileResponse((THIS_DIR / "chat_app.html"), media_type="text/html")


@app.get("/chat_app.ts")
async def main_ts() -> FileResponse:
    """Get the raw typescript code, it's compiled in the browser, forgive me."""
    return FileResponse((THIS_DIR / "chat_app.ts"), media_type="text/plain")


@app.get("/chat/")
async def get_chat(database: Database = Depends(get_db)) -> Response:
    """Get all chat messages."""
    msgs = await database.get_messages()

    # Filter and convert messages, skipping system-only messages
    chat_messages = []
    for m in msgs:
        try:
            chat_msg = to_chat_message(m)
            chat_messages.append(json.dumps(chat_msg).encode("utf-8"))
        except Exception:
            # Skip messages that can't be converted (like system-only messages)
            continue

    return Response(
        b"\n".join(chat_messages),
        media_type="text/plain",
    )


@app.post("/chat/")
async def post_chat(
    prompt: Annotated[str, fastapi.Form()], database: Database = Depends(get_db)
) -> StreamingResponse:
    """Handle new chat messages and stream AI responses."""

    async def stream_messages():
        """Streams new line delimited JSON `Message`s to the client."""
        # Stream the user prompt so it can be displayed immediately
        user_message = create_user_message(prompt)
        yield json.dumps(user_message).encode("utf-8") + b"\n"

        # Get chat history to pass as context to the agent
        messages = await database.get_messages()

        # Run the agent with the user prompt and chat history
        async with agent.run_stream(prompt, message_history=messages) as result:
            async for text in result.stream_output(debounce_by=0.01):
                # Create a ModelResponse for the frontend
                m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                yield json.dumps(to_chat_message(m)).encode("utf-8") + b"\n"

        # Save new messages to the database
        await database.add_messages(result.new_messages_json())

    return StreamingResponse(stream_messages(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", reload=True, reload_dirs=[str(THIS_DIR)])
