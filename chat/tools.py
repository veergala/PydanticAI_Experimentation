"""Simple weather and time tools for the chat app."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx
from pydantic_ai import RunContext

if TYPE_CHECKING:
    from pydantic_ai import Agent


async def get_user_location() -> str:
    """Get user's approximate location based on IP address."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://ip-api.com/json/", timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "success":
                city = data.get("city", "")
                region = data.get("regionName", "")
                country = data.get("country", "")

                if city and region:
                    return f"{city}, {region}"
                elif city:
                    return city
                elif region:
                    return region
                elif country:
                    return country

        return "London"
    except Exception:
        return "London"


def register_tools(agent: "Agent") -> None:
    """Register weather and time tools with the chat agent."""

    @agent.tool
    async def get_current_weather(ctx: RunContext, location: str = "") -> str:
        """Get current weather conditions for any location. If no location provided, uses user's current location."""
        try:
            if not location:
                location = await get_user_location()

            async with httpx.AsyncClient() as client:
                url = f"https://wttr.in/{location}?format=%C+%t+%h+%w"
                response = await client.get(url, timeout=10)
                response.raise_for_status()
                return f"Weather in {location}: {response.text.strip()}"
        except Exception as e:
            return f"Sorry, couldn't get weather for {location}. Error: {str(e)}"

    @agent.tool
    async def get_current_time(ctx: RunContext) -> str:
        """Get the current date and time."""
        now = datetime.now(tz=timezone.utc)
        local_time = now.astimezone()
        return f"Current time: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"