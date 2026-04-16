# harness/scenarios.py
from dataclasses import dataclass
from typing import Callable

@dataclass
class Scenario:
    name: str
    question: str
    tools: list[Callable]
    golden_answer_substring: str  # loose string match for success
    expected_min_tool_calls: int

# --- Tools ---

def get_weather(city: str) -> str:
    """Return weather for a city."""
    fake = {"Tokyo": "sunny, 72F", "Paris": "cloudy, 60F", "Cairo": "hot, 95F"}
    return fake.get(city, "unknown")

def search(query: str) -> str:
    """Search the web."""
    if "python" in query.lower():
        return "Python is a programming language created by Guido van Rossum in 1991."
    return "No results."

def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

def word_count(text: str) -> str:
    """Count words in a string."""
    return str(len(text.split()))

def reverse_string(s: str) -> str:
    """Reverse a string."""
    return s[::-1]

def uppercase(s: str) -> str:
    """Uppercase a string."""
    return s.upper()

def lowercase(s: str) -> str:
    """Lowercase a string."""
    return s.lower()

def length(s: str) -> str:
    """Return length of a string."""
    return str(len(s))

def current_year() -> str:
    """Return the current year."""
    return "2026"

def capital_of(country: str) -> str:
    """Return the capital city of a country."""
    fake = {"France": "Paris", "Japan": "Tokyo", "Egypt": "Cairo"}
    return fake.get(country, "unknown")


S1 = Scenario(
    name="s1_single_tool",
    question="What's the weather in Tokyo?",
    tools=[get_weather],
    golden_answer_substring="sunny",
    expected_min_tool_calls=1,
)

S3 = Scenario(
    name="s3_three_tools",
    question="Search for who created Python, then tell me the year plus 10.",
    tools=[search, calculator, word_count],
    golden_answer_substring="2001",
    expected_min_tool_calls=2,
)

S10 = Scenario(
    name="s10_ten_tools",
    question=(
        "What is the capital of France? Then check its weather. "
        "Then return the uppercase of the weather description."
    ),
    tools=[
        get_weather, search, calculator, word_count, reverse_string,
        uppercase, lowercase, length, current_year, capital_of,
    ],
    golden_answer_substring="CLOUDY",
    expected_min_tool_calls=3,
)

ALL_SCENARIOS = {"s1": S1, "s3": S3, "s10": S10}
