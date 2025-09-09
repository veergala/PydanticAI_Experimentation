import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def roulette_wheel(square: int, winning_number: int) -> str:
    """Check if the square is a winner"""
    return "winner" if square == winning_number else "loser"


def run_roulette_game(user_message: str, winning_number: int) -> bool:
    """
    Run a roulette game using OpenAI API with function calling
    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "roulette_wheel",
                "description": "Check if the square is a winner",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "square": {
                            "type": "integer",
                            "description": "The roulette square number the customer is betting on",
                        }
                    },
                    "required": ["square"],
                },
            },
        }
    ]

    system_prompt = (
        "Use the `roulette_wheel` function to see if the "
        "customer has won based on the number they provide. "
        "Extract the number from their message and check if they won. "
        "Return true if they won, false if they lost."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, tools=tools, tool_choice="auto"
    )

    message = response.choices[0].message
    messages.append(message)

    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.name == "roulette_wheel":
                function_args = json.loads(tool_call.function.arguments)
                square = function_args["square"]

                result = roulette_wheel(square, winning_number)
                return result == "winner"

    return False


if __name__ == "__main__":
    success_number = 18

    result1 = run_roulette_game("Put my money on square eighteen", success_number)
    print(f"Result 1: {result1}")

    result2 = run_roulette_game("I bet five is the winner", success_number)
    print(f"Result 2: {result2}")
