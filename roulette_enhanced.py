import random
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

load_dotenv()

# Roulette wheel setup
ROULETTE_NUMBERS = list(range(37)) + [37]  # 0-36 + 37 (representing 00)
RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BLACK_NUMBERS = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}


@dataclass
class GameState:
    balance: int = 1000
    winning_number: Optional[int] = None
    last_spin_result: Optional[str] = None


def get_number_color(number: int) -> str:
    """Get the color of a roulette number"""
    if number in [0, 37]:  # 0 and 00
        return "green"
    elif number in RED_NUMBERS:
        return "red"
    elif number in BLACK_NUMBERS:
        return "black"
    return "unknown"


def format_number(number: int) -> str:
    """Format number for display (37 becomes '00')"""
    return "00" if number == 37 else str(number)


def calculate_payout(
    bet_type: str, bet_amount: int, winning_number: int, bet_value: any
) -> int:
    """Calculate payout based on bet type and winning number"""
    if bet_type == "straight":  # Single number bet
        return bet_amount * 35 if winning_number == bet_value else 0
    elif bet_type == "color":
        winning_color = get_number_color(winning_number)
        return bet_amount * 1 if winning_color == bet_value else 0
    elif bet_type == "odd_even":
        if winning_number in [0, 37]:  # 0 and 00 lose
            return 0
        is_odd = winning_number % 2 == 1
        bet_is_odd = bet_value == "odd"
        return bet_amount * 1 if is_odd == bet_is_odd else 0
    elif bet_type == "high_low":
        if winning_number in [0, 37]:  # 0 and 00 lose
            return 0
        is_high = winning_number >= 19
        bet_is_high = bet_value == "high"
        return bet_amount * 1 if is_high == bet_is_high else 0
    return 0


model_choice = "openai:gpt-4o"

roulette_agent = Agent(
    model=model_choice,
    deps_type=GameState,
    system_prompt=(
        "You are a roulette dealer. Help players place bets and spin the wheel. "
        "When players say 'all in', 'put it all', 'everything', or similar, "
        "use their entire current balance as the bet amount. "
        "Always process bets and spin in one seamless action. "
        "Be friendly and concise. Available bet types: "
        "straight (single number), color (red/black), odd_even, high_low."
    ),
)


@roulette_agent.tool
async def spin_wheel(ctx: RunContext[GameState]) -> str:
    """Spin the roulette wheel and return the winning number"""
    winning_number = random.choice(ROULETTE_NUMBERS)
    ctx.deps.winning_number = winning_number
    color = get_number_color(winning_number)
    formatted_number = format_number(winning_number)

    result = f"🎰 The wheel spins... and lands on {formatted_number} ({color})!"
    ctx.deps.last_spin_result = result
    return result


@roulette_agent.tool
async def get_current_balance(ctx: RunContext[GameState]) -> int:
    """Get the player's current balance for betting calculations"""
    return ctx.deps.balance


@roulette_agent.tool
async def place_bet(
    ctx: RunContext[GameState], bet_type: str, bet_value: str, amount: int
) -> str:
    """Place a bet on the roulette table"""
    if amount <= 0:
        return "❌ Bet amount must be positive!"

    if amount > ctx.deps.balance:
        return f"❌ Insufficient funds! Your balance is ${ctx.deps.balance}"

    # Validate bet types
    valid_bet_types = ["straight", "color", "odd_even", "high_low"]
    if bet_type not in valid_bet_types:
        return f"❌ Invalid bet type! Valid types: {', '.join(valid_bet_types)}"

    # Validate bet values based on type
    if bet_type == "straight":
        try:
            num = int(bet_value) if bet_value != "00" else 37
            if num not in ROULETTE_NUMBERS:
                return "❌ Invalid number! Choose 0, 00, or 1-36"
        except ValueError:
            return "❌ Invalid number format!"
    elif bet_type == "color" and bet_value not in ["red", "black"]:
        return "❌ Invalid color! Choose 'red' or 'black'"
    elif bet_type == "odd_even" and bet_value not in ["odd", "even"]:
        return "❌ Invalid choice! Choose 'odd' or 'even'"
    elif bet_type == "high_low" and bet_value not in ["high", "low"]:
        return "❌ Invalid choice! Choose 'high' (19-36) or 'low' (1-18)"

    return f"✅ Bet placed: ${amount} on {bet_type} {bet_value}"


@roulette_agent.tool
async def check_results(
    ctx: RunContext[GameState], bet_type: str, bet_value: str, bet_amount: int
) -> str:
    """Check if the player won their bet and calculate payout"""
    if ctx.deps.winning_number is None:
        return "❌ No spin result available! Spin the wheel first."

    # Convert bet_value for calculation
    calc_bet_value = bet_value
    if bet_type == "straight":
        calc_bet_value = int(bet_value) if bet_value != "00" else 37

    payout = calculate_payout(
        bet_type, bet_amount, ctx.deps.winning_number, calc_bet_value
    )

    if payout > 0:
        ctx.deps.balance += payout
        return f"🎉 Winner! You won ${payout}! New balance: ${ctx.deps.balance}"
    else:
        ctx.deps.balance -= bet_amount
        if ctx.deps.balance <= 0:
            return (
                f"💀 BUST! You lost ${bet_amount} and are completely broke! Balance: $0"
            )
        else:
            return f"💔 Sorry, you lost ${bet_amount}. New balance: ${ctx.deps.balance}"


@roulette_agent.tool
async def get_balance(ctx: RunContext[GameState]) -> str:
    """Get the current player balance"""
    return f"💰 Your current balance: ${ctx.deps.balance}"


@roulette_agent.tool
async def get_game_rules(ctx: RunContext[GameState]) -> str:
    """Get the roulette game rules and payouts"""
    rules = """
🎰 ROULETTE RULES & PAYOUTS:

BET TYPES:
• Straight (single number): Bet on one number (0, 00, 1-36) - Pays 35:1
• Color: Bet on red or black - Pays 1:1
• Odd/Even: Bet on odd or even numbers - Pays 1:1
• High/Low: Bet on low (1-18) or high (19-36) - Pays 1:1

NOTES:
• 0 and 00 are green and win only straight bets
• Color, odd/even, and high/low bets lose on 0 and 00
• You start with $1000
    """
    return rules.strip()


def main():
    """Simple game loop"""
    print("🎰 Welcome to Enhanced PydanticAI Roulette! 🎰")
    print("Type your bets in natural language, like:")
    print("- 'I bet $50 on red'")
    print("- 'Put $25 on number 17'")
    print("- 'I want to bet $100 on odd numbers'")
    print("- Type 'rules' for game rules")
    print("- Type 'balance' to check your money")
    print("- Type 'quit' to exit\n")

    game_state = GameState()

    while True:
        user_input = input("🎯 Place your bet: ").strip()

        if user_input.lower() == "quit":
            print(f"Thanks for playing! Final balance: ${game_state.balance}")
            break
        elif user_input.lower() == "rules":
            result = roulette_agent.run_sync("Show me the game rules", deps=game_state)
            print(result.output)
            continue
        elif user_input.lower() == "balance":
            result = roulette_agent.run_sync("What's my balance?", deps=game_state)
            print(result.output)
            continue

        try:
            result = roulette_agent.run_sync(user_input, deps=game_state)
            print(result.output)

            # Check if player has lost all money
            if game_state.balance <= 0:
                print("\n💀 GAME OVER! You've lost all your money!")
                print("🎰 Thanks for playing Enhanced PydanticAI Roulette!")
                print("💡 Better luck next time - the house always wins in the end!")
                break

        except Exception as e:
            print(f"❌ Error: {e}")
            print("Try rephrasing your bet or type 'rules' for help")


if __name__ == "__main__":
    main()
