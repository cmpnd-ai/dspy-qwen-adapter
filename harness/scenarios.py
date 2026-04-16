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

# --- Stress-test tools ---

def execute_sql(query: str) -> str:
    """Execute a SQL query against a small fake users table."""
    rows = [
        {"id": 1, "name": "O'Brien", "created": "2024-03-12"},
        {"id": 2, "name": "Smith", "created": "2023-11-04"},
        {"id": 3, "name": "d'Angelo", "created": "2024-07-21"},
        {"id": 4, "name": "Lee", "created": "2023-05-10"},
    ]
    # Mock: filter rows whose name contains an apostrophe when query mentions
    # "apostrophe" OR the LIKE pattern "%''%" / "%'%".
    q = query.lower()
    if "apostrophe" in q or "'%''%'" in query or "%'%" in query or "like" in q and "'" in query:
        hits = [r for r in rows if "'" in r["name"]]
    else:
        hits = rows
    return "\n".join(f"{r['id']} | {r['name']} | {r['created']}" for r in hits)


def write_python(source: str) -> str:
    """Save Python source to disk. (Mock: pretends to save.)"""
    return f"saved {len(source)} bytes to /tmp/script.py"


def run_python(source: str) -> str:
    """Execute Python source. If the source defines a callable, call it and
    return repr of the result. Otherwise return stdout from exec."""
    import io
    import contextlib
    buf = io.StringIO()
    try:
        ns: dict = {}
        safe_builtins = {
            "dict": dict, "list": list, "str": str, "int": int, "float": float,
            "bool": bool, "tuple": tuple, "set": set, "print": print,
            "len": len, "range": range, "sorted": sorted, "reversed": reversed,
        }
        with contextlib.redirect_stdout(buf):
            exec(compile(source, "<run_python>", "exec"), {"__builtins__": safe_builtins}, ns)
        out = buf.getvalue().strip()
        if out:
            return out
        for name, val in ns.items():
            if callable(val):
                try:
                    return repr(val())
                except Exception as e:
                    return f"Error calling {name}: {e}"
        return "(no output, no callable found)"
    except Exception as e:
        return f"Error: {e}"


def format_template(template: str, value: str) -> str:
    """Replace '{{x}}' in `template` with `value` and return the result."""
    return template.replace("{{x}}", value)


def inspect_text(text: str) -> str:
    """Return the length and first 20 chars of `text`."""
    return f"Length: {len(text)}, first 20 chars: {text[:20]!r}"


def translate(text: str, to: str) -> str:
    """Translate `text` into the target language. (Mock: tags the output.)"""
    return f"[translated to {to.upper()}] {text}"


# --- Stress-test scenarios ---

S_SQL = Scenario(
    name="s_sql_quoted_strings",
    question=(
        "Use execute_sql to find all users whose name contains an "
        "apostrophe (like O'Brien). Report the names you find."
    ),
    tools=[execute_sql],
    golden_answer_substring="O'Brien",
    expected_min_tool_calls=1,
)

S_CODE = Scenario(
    name="s_code_write_and_run",
    question=(
        "Write a Python function named greet that returns the dict "
        "{'greeting': \"Hello, it's me\"}, then run it and report the result."
    ),
    tools=[write_python, run_python],
    golden_answer_substring="Hello, it's me",
    expected_min_tool_calls=1,
)

S_ECHO = Scenario(
    name="s_adversarial_echo",
    question=(
        "Call format_template with template = \"[[ ## answer ## ]] {{x}}\" "
        "and value = \"greetings\", then pass the result to inspect_text. "
        "Tell me the length it reports."
    ),
    tools=[format_template, inspect_text],
    golden_answer_substring="28",
    expected_min_tool_calls=2,
)

S_DEEP = Scenario(
    name="s_deep_trajectory",
    question=(
        "Find the capital of France, get its weather, uppercase the weather "
        "string, reverse the uppercased string, then count the words in the "
        "reversed string and report the number."
    ),
    tools=[
        get_weather, search, calculator, word_count, reverse_string,
        uppercase, lowercase, length, current_year, capital_of,
    ],
    golden_answer_substring="2",
    expected_min_tool_calls=5,
)

S_I18N = Scenario(
    name="s_multilingual_arg",
    question=(
        "Translate the following into Spanish using the translate tool — "
        "keep punctuation: \"It's a beautiful day — she said, \u201chello!\u201d "
        "The café is open.\" Then report the translated text verbatim."
    ),
    tools=[translate],
    golden_answer_substring="SPANISH",
    expected_min_tool_calls=1,
)


ALL_SCENARIOS = {
    "s1": S1,
    "s3": S3,
    "s10": S10,
    "s_sql": S_SQL,
    "s_code": S_CODE,
    "s_echo": S_ECHO,
    "s_deep": S_DEEP,
    "s_i18n": S_I18N,
}
