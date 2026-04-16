# Qwen35Adapter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and ship `dspy-qwen35-adapter`, a PyPI package exposing `Qwen35Adapter` — a text-native DSPy adapter that makes `dspy.ReAct` reliable with Qwen 3.5 35B A3B on any OpenAI-compatible endpoint.

**Architecture:** Subclass `dspy.adapters.base.Adapter`. Hardcode `use_native_function_calling=False`; never pass `tools=[]` to the server. Prompt Qwen in its trained XML format (`<function=NAME><parameter=K>V</parameter></function>`), parse tool calls out of `message.content` ourselves, strip `<think>` blocks, ignore `finish_reason`. Keep parsing in pure functions (`parsing.py`) so it can be unit-tested against a fixture corpus of real failure modes.

**Tech Stack:** Python 3.10+, DSPy (as git submodule, editable install), LiteLLM (transitive via DSPy), `json-repair` for lenient JSON decoding, `pytest` for tests, LM Studio as the reference local server.

**Prerequisites:**
- LM Studio running on `http://localhost:1234` with Qwen 3.5 35B A3B loaded (Unsloth GGUF recommended per `qwen-35-toolnotes-from-reddit.md`).
- Read `docs/plans/2026-04-16-qwen35-adapter-design.md` first — this plan assumes you understand the design.
- Working directory: repo root (`/Users/dbreunig/Development/cmpnd/dspy-qwen35-adapter`). No worktree is in use.

**Global rules:**
- TDD throughout: write the failing test, run it to confirm it fails, implement the minimum to pass, run it to confirm it passes, commit. One test at a time.
- Commit after every green test. Messages start with `feat:`, `test:`, `refactor:`, `docs:`, or `chore:`.
- Do not add features beyond the task. No speculative generality. YAGNI.
- No LM calls in `tests/`. Those are pure + mocked. LM calls live in `harness/`.

---

## Task 1: Initialize Python package

**Files:**
- Create: `pyproject.toml`
- Create: `dspy_qwen35_adapter/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/traces/.gitkeep`
- Create: `harness/__init__.py`
- Create: `harness/traces/.gitkeep`
- Create: `harness/results/.gitkeep`
- Modify: `.gitignore`

**Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "dspy-qwen35-adapter"
version = "0.1.0"
description = "DSPy adapter for Qwen 3.5 tool calling on OpenAI-compatible local servers"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
    "dspy>=2.5",
    "json-repair>=0.25",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov"]

[tool.hatch.build.targets.wheel]
packages = ["dspy_qwen35_adapter"]

[tool.hatch.build.targets.sdist]
exclude = ["tests/", "harness/", "docs/", "dspy/"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Write `dspy_qwen35_adapter/__init__.py`**

```python
from dspy_qwen35_adapter.adapter import Qwen35Adapter

__all__ = ["Qwen35Adapter"]
__version__ = "0.1.0"
```

Note: this import will fail until Task 8 lands. That's fine — Task 1 only verifies the package skeleton exists; we don't run it yet.

**Step 3: Create empty module init files and keep files**

```bash
mkdir -p dspy_qwen35_adapter tests/fixtures/traces harness/traces harness/results
touch tests/__init__.py harness/__init__.py
touch tests/fixtures/traces/.gitkeep harness/traces/.gitkeep harness/results/.gitkeep
```

**Step 4: Extend `.gitignore`**

Append:

```
__pycache__/
*.pyc
.pytest_cache/
.venv/
*.egg-info/
dist/
build/
harness/results/*.csv
harness/traces/*.txt
!harness/results/.gitkeep
!harness/traces/.gitkeep
```

**Step 5: Install the package in editable mode + the DSPy submodule**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ./dspy
pip install -e '.[dev]'
python -c "import dspy; import dspy_qwen35_adapter; print('ok')"
```

Expected output: `ImportError` from the `Qwen35Adapter` import — that's fine. Verify `import dspy` works before proceeding.

**Step 6: Commit**

```bash
git add pyproject.toml dspy_qwen35_adapter/ tests/ harness/ .gitignore
git commit -m "chore: scaffold Python package, tests, harness"
```

---

## Task 2: Implement `strip_think` (TDD)

**Files:**
- Create: `dspy_qwen35_adapter/parsing.py`
- Create: `tests/test_parsing.py`

**Step 1: Write the failing tests**

```python
# tests/test_parsing.py
from dspy_qwen35_adapter.parsing import strip_think


def test_strip_think_removes_balanced_block():
    text = "<think>reasoning</think>Hello world"
    assert strip_think(text) == "Hello world"


def test_strip_think_removes_multiple_blocks():
    text = "<think>a</think>hi<think>b</think>bye"
    assert strip_think(text) == "hibye"


def test_strip_think_handles_orphan_closer():
    text = "</think>Hello world"
    assert strip_think(text) == "Hello world"


def test_strip_think_handles_unclosed_block():
    text = "Before<think>unclosed..."
    assert strip_think(text) == "Before"


def test_strip_think_passthrough_when_no_tags():
    text = "Plain text answer"
    assert strip_think(text) == "Plain text answer"


def test_strip_think_preserves_internal_whitespace():
    text = "<think>r</think>\n\nHello"
    assert strip_think(text) == "Hello"
```

**Step 2: Run tests to confirm failure**

Run: `pytest tests/test_parsing.py -v`
Expected: `ImportError: cannot import name 'strip_think'`

**Step 3: Implement `strip_think`**

```python
# dspy_qwen35_adapter/parsing.py
import re

_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>")
_ORPHAN_CLOSER = re.compile(r"^\s*</think>\s*")
_UNCLOSED_THINK = re.compile(r"<think>[\s\S]*$")


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks, orphan </think> openers, and
    unclosed <think> tails. Whitespace around removed regions is trimmed."""
    text = _ORPHAN_CLOSER.sub("", text)
    text = _THINK_BLOCK.sub("", text)
    text = _UNCLOSED_THINK.sub("", text)
    return text.strip()
```

**Step 4: Run tests to confirm pass**

Run: `pytest tests/test_parsing.py -v`
Expected: 6 passed.

**Step 5: Commit**

```bash
git add dspy_qwen35_adapter/parsing.py tests/test_parsing.py
git commit -m "feat: implement strip_think for Qwen think-tag cleanup"
```

---

## Task 3: Implement `extract_tool_call` (TDD)

**Files:**
- Modify: `dspy_qwen35_adapter/parsing.py`
- Modify: `tests/test_parsing.py`

**Step 1: Write failing tests**

Append to `tests/test_parsing.py`:

```python
from dspy_qwen35_adapter.parsing import extract_tool_call


def test_extract_tool_call_single_string_param():
    text = "<function=get_weather><parameter=city>Tokyo</parameter></function>"
    assert extract_tool_call(text) == ("get_weather", {"city": "Tokyo"})


def test_extract_tool_call_multiple_params():
    text = (
        "<function=search>"
        "<parameter=query>weather</parameter>"
        "<parameter=limit>5</parameter>"
        "</function>"
    )
    assert extract_tool_call(text) == ("search", {"query": "weather", "limit": 5})


def test_extract_tool_call_json_value_decoded():
    text = (
        '<function=run>'
        '<parameter=config>{"key": "value"}</parameter>'
        '</function>'
    )
    assert extract_tool_call(text) == ("run", {"config": {"key": "value"}})


def test_extract_tool_call_no_params():
    text = "<function=finish></function>"
    assert extract_tool_call(text) == ("finish", {})


def test_extract_tool_call_returns_none_when_absent():
    assert extract_tool_call("plain answer text") is None


def test_extract_tool_call_takes_first_when_multiple():
    text = (
        "<function=a><parameter=x>1</parameter></function>"
        "<function=b><parameter=y>2</parameter></function>"
    )
    assert extract_tool_call(text) == ("a", {"x": 1})


def test_extract_tool_call_malformed_returns_none():
    text = "<function=broken><parameter=x>unclosed"
    assert extract_tool_call(text) is None
```

**Step 2: Run tests to confirm failure**

Run: `pytest tests/test_parsing.py -v`
Expected: `ImportError: cannot import name 'extract_tool_call'`.

**Step 3: Implement**

Append to `dspy_qwen35_adapter/parsing.py`:

```python
from typing import Any

try:
    import json_repair
    _json_loads = json_repair.loads
except ImportError:  # pragma: no cover
    import json
    _json_loads = json.loads

_FUNCTION_BLOCK = re.compile(r"<function=(\S+?)>([\s\S]*?)</function>")
_PARAMETER_BLOCK = re.compile(r"<parameter=(\S+?)>([\s\S]*?)</parameter>")


def _decode_value(raw: str) -> Any:
    raw = raw.strip()
    if raw == "":
        return ""
    try:
        return _json_loads(raw)
    except Exception:
        return raw


def extract_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Return the first (tool_name, args) found in the text, or None.
    Parameter values are JSON-decoded when possible, else returned as strings."""
    fn_match = _FUNCTION_BLOCK.search(text)
    if not fn_match:
        return None
    name = fn_match.group(1)
    body = fn_match.group(2)
    args: dict[str, Any] = {}
    for p in _PARAMETER_BLOCK.finditer(body):
        args[p.group(1)] = _decode_value(p.group(2))
    return name, args
```

**Step 4: Run tests to confirm pass**

Run: `pytest tests/test_parsing.py -v`
Expected: all passing.

**Step 5: Commit**

```bash
git add dspy_qwen35_adapter/parsing.py tests/test_parsing.py
git commit -m "feat: implement extract_tool_call for Qwen XML tool calls"
```

---

## Task 4: Implement `split_thought_and_call` (TDD)

**Files:**
- Modify: `dspy_qwen35_adapter/parsing.py`
- Modify: `tests/test_parsing.py`

**Step 1: Failing tests**

```python
from dspy_qwen35_adapter.parsing import split_thought_and_call


def test_split_with_thought_before_call():
    text = (
        "I should check the weather.\n"
        "<function=get_weather><parameter=city>Tokyo</parameter></function>"
    )
    thought, call = split_thought_and_call(text)
    assert thought == "I should check the weather."
    assert call == ("get_weather", {"city": "Tokyo"})


def test_split_with_no_call():
    thought, call = split_thought_and_call("Just a plain answer.")
    assert thought == "Just a plain answer."
    assert call is None


def test_split_with_call_only_no_thought():
    text = "<function=finish></function>"
    thought, call = split_thought_and_call(text)
    assert thought == ""
    assert call == ("finish", {})


def test_split_strips_surrounding_whitespace_in_thought():
    text = "\n\n  reasoning  \n<function=f></function>"
    thought, _ = split_thought_and_call(text)
    assert thought == "reasoning"
```

**Step 2: Run — expect `ImportError`.**

**Step 3: Implement**

```python
def split_thought_and_call(
    text: str,
) -> tuple[str, tuple[str, dict[str, Any]] | None]:
    """Split content into (thought, tool_call). Thought is text before the
    first <function=...> tag; tool_call is the first parsed function block."""
    fn_match = _FUNCTION_BLOCK.search(text)
    if not fn_match:
        return text.strip(), None
    thought = text[: fn_match.start()].strip()
    name = fn_match.group(1)
    args: dict[str, Any] = {}
    for p in _PARAMETER_BLOCK.finditer(fn_match.group(2)):
        args[p.group(1)] = _decode_value(p.group(2))
    return thought, (name, args)
```

**Step 4: Run — expect pass.**

**Step 5: Commit**

```bash
git commit -am "feat: implement split_thought_and_call"
```

---

## Task 5: Implement `coerce_args_to_schema` (TDD)

**Files:**
- Modify: `dspy_qwen35_adapter/parsing.py`
- Modify: `tests/test_parsing.py`

**Step 1: Failing tests**

```python
from dspy_qwen35_adapter.parsing import coerce_args_to_schema


def test_coerce_string_to_int():
    schema = {"count": {"type": "integer"}}
    assert coerce_args_to_schema({"count": "5"}, schema) == {"count": 5}


def test_coerce_string_to_float():
    schema = {"ratio": {"type": "number"}}
    assert coerce_args_to_schema({"ratio": "1.5"}, schema) == {"ratio": 1.5}


def test_coerce_string_to_bool():
    schema = {"enabled": {"type": "boolean"}}
    assert coerce_args_to_schema({"enabled": "true"}, schema) == {"enabled": True}
    assert coerce_args_to_schema({"enabled": "false"}, schema) == {"enabled": False}


def test_coerce_passes_through_on_failure():
    schema = {"count": {"type": "integer"}}
    assert coerce_args_to_schema({"count": "not a number"}, schema) == {"count": "not a number"}


def test_coerce_ignores_unknown_keys():
    schema = {"a": {"type": "integer"}}
    assert coerce_args_to_schema({"b": "5"}, schema) == {"b": "5"}
```

**Step 2: Run — expect fail.**

**Step 3: Implement**

```python
def coerce_args_to_schema(
    args: dict[str, Any], schema: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Best-effort coerce string args to integer/number/boolean per a simple
    JSON-schema-like map. Leaves values unchanged on any failure or when no
    type hint is present."""
    out = dict(args)
    for key, value in args.items():
        spec = schema.get(key)
        if not spec or not isinstance(value, str):
            continue
        kind = spec.get("type")
        try:
            if kind == "integer":
                out[key] = int(value)
            elif kind == "number":
                out[key] = float(value)
            elif kind == "boolean":
                lowered = value.strip().lower()
                if lowered in {"true", "false"}:
                    out[key] = lowered == "true"
        except (ValueError, TypeError):
            pass
    return out
```

**Step 4: Run — expect pass.**

**Step 5: Commit**

```bash
git commit -am "feat: implement coerce_args_to_schema"
```

---

## Task 6: Seed fixture corpus for parsing regression tests

**Files:**
- Create: `tests/fixtures/traces/01-clean-single-call.txt` (+ `.json`)
- Create: `tests/fixtures/traces/02-think-leaked.txt` (+ `.json`)
- Create: `tests/fixtures/traces/03-unclosed-think.txt` (+ `.json`)
- Create: `tests/fixtures/traces/04-xml-with-preamble.txt` (+ `.json`)
- Create: `tests/fixtures/traces/05-no-tool-plain-text.txt` (+ `.json`)
- Create: `tests/fixtures/traces/06-malformed-missing-close.txt` (+ `.json`)
- Create: `tests/fixtures/traces/07-two-calls-takes-first.txt` (+ `.json`)
- Create: `tests/fixtures/traces/08-args-with-json.txt` (+ `.json`)
- Modify: `tests/test_parsing.py`

**Step 1: Create fixture pairs**

Example pair — `01-clean-single-call.txt`:
```
I'll check the weather for Tokyo.
<function=get_weather><parameter=city>Tokyo</parameter></function>
```

`01-clean-single-call.json`:
```json
{"thought": "I'll check the weather for Tokyo.", "tool_name": "get_weather", "tool_args": {"city": "Tokyo"}}
```

Create all 8 pairs. Use the Reddit-documented failure modes as content seeds. `no_tool` fixtures should set `"tool_name": null` and `"tool_args": null` in the expected JSON.

**Step 2: Write the discovery-based test**

```python
import json
from pathlib import Path
import pytest
from dspy_qwen35_adapter.parsing import strip_think, split_thought_and_call

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "traces"


def _fixture_pairs():
    for txt_path in sorted(FIXTURE_DIR.glob("*.txt")):
        json_path = txt_path.with_suffix(".json")
        if json_path.exists():
            yield pytest.param(txt_path, json_path, id=txt_path.stem)


@pytest.mark.parametrize("txt_path,json_path", list(_fixture_pairs()))
def test_fixture_parses_to_expected(txt_path, json_path):
    raw = txt_path.read_text()
    expected = json.loads(json_path.read_text())
    cleaned = strip_think(raw)
    thought, call = split_thought_and_call(cleaned)
    assert thought == expected["thought"]
    if expected.get("tool_name") is None:
        assert call is None
    else:
        assert call is not None
        assert call[0] == expected["tool_name"]
        assert call[1] == expected["tool_args"]
```

**Step 3: Run — all 8 fixtures should pass.**

Run: `pytest tests/test_parsing.py -v`
Expected: all tests green.

**Step 4: Commit**

```bash
git add tests/fixtures/ tests/test_parsing.py
git commit -m "test: seed fixture corpus for Qwen parsing regressions"
```

---

## Task 7: Implement `build_system_prompt` (TDD)

**Files:**
- Create: `dspy_qwen35_adapter/prompts.py`
- Create: `tests/test_prompts.py`

**Step 1: Failing tests**

```python
# tests/test_prompts.py
import dspy
from dspy.adapters.types.tool import Tool
from dspy_qwen35_adapter.prompts import build_system_prompt


def _weather_tool():
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        return ""
    return Tool(get_weather)


def test_prompt_includes_tools_block():
    tools = [_weather_tool()]
    prompt = build_system_prompt("Answer the question.", tools)
    assert "<tools>" in prompt
    assert "</tools>" in prompt
    assert "get_weather" in prompt


def test_prompt_includes_tool_description():
    prompt = build_system_prompt("T", [_weather_tool()])
    assert "Get current weather for a city." in prompt


def test_prompt_includes_parameter_schema():
    prompt = build_system_prompt("T", [_weather_tool()])
    assert '"city"' in prompt


def test_prompt_includes_exemplar():
    prompt = build_system_prompt("T", [_weather_tool()])
    assert "<function=" in prompt
    assert "<parameter=" in prompt


def test_prompt_includes_task_instructions():
    prompt = build_system_prompt("Answer the question thoroughly.", [_weather_tool()])
    assert "Answer the question thoroughly." in prompt


def test_prompt_empty_tools_omits_tools_block():
    prompt = build_system_prompt("T", [])
    assert "<tools>" not in prompt
```

**Step 2: Run — expect `ImportError`.**

**Step 3: Implement**

```python
# dspy_qwen35_adapter/prompts.py
import json
from typing import Iterable
from dspy.adapters.types.tool import Tool


_EXEMPLAR = (
    "Example of a tool call:\n"
    "<function=example_tool>"
    "<parameter=arg1>value1</parameter>"
    "<parameter=arg2>value2</parameter>"
    "</function>"
)


def _tool_to_xml(tool: Tool) -> str:
    schema = tool.args or {}
    return (
        f'  <tool name="{tool.name}">\n'
        f'    <description>{tool.desc or ""}</description>\n'
        f'    <parameters>{json.dumps(schema)}</parameters>\n'
        f'  </tool>'
    )


def build_system_prompt(task_description: str, tools: Iterable[Tool]) -> str:
    """Build a system prompt that keeps Qwen 3.5 in-distribution for tool use.
    Emits a <tools> block and a one-shot XML exemplar only when tools are
    provided."""
    tool_list = list(tools)
    sections = [task_description.strip()] if task_description else []
    if tool_list:
        block = "<tools>\n" + "\n".join(_tool_to_xml(t) for t in tool_list) + "\n</tools>"
        sections.append(block)
        sections.append(_EXEMPLAR)
        sections.append(
            "Emit exactly one <function=...>...</function> per turn, or answer in plain text when done."
        )
    return "\n\n".join(sections)
```

**Step 4: Run — expect pass.**

**Step 5: Commit**

```bash
git add dspy_qwen35_adapter/prompts.py tests/test_prompts.py
git commit -m "feat: implement build_system_prompt for Qwen tool-use"
```

---

## Task 8: Skeleton `Qwen35Adapter` class — `__init__` and `parse` happy path (TDD)

**Files:**
- Create: `dspy_qwen35_adapter/adapter.py`
- Create: `tests/test_adapter_integration.py`

**Step 1: Failing tests**

```python
# tests/test_adapter_integration.py
import dspy
import pytest
from dspy.adapters.types.tool import Tool
from dspy_qwen35_adapter import Qwen35Adapter


def _sample_signature_with_tools():
    class Sig(dspy.Signature):
        """Answer the question."""
        question: str = dspy.InputField()
        tools: list[Tool] = dspy.InputField()
        trajectory: str = dspy.InputField()
        next_thought: str = dspy.OutputField()
        next_tool_name: str = dspy.OutputField()
        next_tool_args: dict = dspy.OutputField()
    return Sig


def test_adapter_instantiates():
    a = Qwen35Adapter()
    assert a.use_native_function_calling is False
    assert a.strict_parse is False


def test_adapter_accepts_strict_parse_flag():
    a = Qwen35Adapter(strict_parse=True)
    assert a.strict_parse is True


def test_parse_happy_path_returns_react_fields():
    a = Qwen35Adapter()
    sig = _sample_signature_with_tools()
    completion = (
        "I should check.\n"
        "<function=get_weather><parameter=city>Tokyo</parameter></function>"
    )
    out = a.parse(sig, completion)
    assert out["next_thought"] == "I should check."
    assert out["next_tool_name"] == "get_weather"
    assert out["next_tool_args"] == {"city": "Tokyo"}


def test_parse_strips_think_before_splitting():
    a = Qwen35Adapter()
    sig = _sample_signature_with_tools()
    completion = (
        "<think>internal</think>"
        "I should check.\n"
        "<function=get_weather><parameter=city>Tokyo</parameter></function>"
    )
    out = a.parse(sig, completion)
    assert "internal" not in out["next_thought"]
    assert out["next_thought"] == "I should check."
```

**Step 2: Run — expect fail.**

**Step 3: Implement**

```python
# dspy_qwen35_adapter/adapter.py
import logging
from typing import Any

from dspy.adapters.base import Adapter
from dspy.adapters.types.tool import Tool
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback

from dspy_qwen35_adapter.parsing import (
    strip_think,
    split_thought_and_call,
    coerce_args_to_schema,
)

logger = logging.getLogger(__name__)

REACT_TOOL_FIELDS = {"next_thought", "next_tool_name", "next_tool_args"}


class Qwen35Adapter(Adapter):
    """Text-native DSPy adapter for Qwen 3.5 tool calling.

    Bypasses the inference server's tool-call parser entirely. Prompts Qwen
    in its trained XML format and parses tool calls out of the assistant's
    content directly.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        native_response_types: list[type] | None = None,
        strict_parse: bool = False,
    ):
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=False,
            native_response_types=native_response_types,
        )
        self.strict_parse = strict_parse

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        cleaned = strip_think(completion)
        thought, call = split_thought_and_call(cleaned)

        has_react_fields = REACT_TOOL_FIELDS.issubset(signature.output_fields.keys())

        if has_react_fields:
            if call is None:
                if self.strict_parse:
                    from dspy.utils.exceptions import AdapterParseError
                    raise AdapterParseError(
                        adapter_name="Qwen35Adapter",
                        signature=signature,
                        lm_response=completion,
                        message="No <function=...> tool call found.",
                    )
                return {
                    "next_thought": thought,
                    "next_tool_name": "finish",
                    "next_tool_args": {},
                }
            name, args = call
            return {
                "next_thought": thought,
                "next_tool_name": name,
                "next_tool_args": args,
            }

        # Non-ReAct signatures fall through to the XML/delimiter fallback
        # added in Task 10. For now, return thought as best-effort.
        return {list(signature.output_fields.keys())[0]: cleaned}
```

**Step 4: Run — expect pass.**

**Step 5: Commit**

```bash
git add dspy_qwen35_adapter/adapter.py tests/test_adapter_integration.py
git commit -m "feat: Qwen35Adapter skeleton with parse() ReAct path"
```

---

## Task 9: Adapter `format_*` methods (TDD)

**Files:**
- Modify: `dspy_qwen35_adapter/adapter.py`
- Modify: `tests/test_adapter_integration.py`

**Step 1: Failing tests** — append to `test_adapter_integration.py`:

```python
def test_format_produces_system_with_tools_block():
    a = Qwen35Adapter()
    sig = _sample_signature_with_tools()

    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "sunny"

    messages = a.format(
        signature=sig,
        demos=[],
        inputs={
            "question": "What's in Tokyo?",
            "tools": [Tool(get_weather)],
            "trajectory": "",
        },
    )
    system = next(m for m in messages if m["role"] == "system")
    assert "<tools>" in system["content"]
    assert "get_weather" in system["content"]


def test_format_assistant_message_replays_qwen_xml():
    a = Qwen35Adapter()
    sig = _sample_signature_with_tools()
    content = a.format_assistant_message_content(
        sig,
        outputs={
            "next_thought": "check weather",
            "next_tool_name": "get_weather",
            "next_tool_args": {"city": "Tokyo"},
        },
    )
    assert "<function=get_weather>" in content
    assert "<parameter=city>Tokyo</parameter>" in content
    assert "check weather" in content
```

**Step 2: Run — expect fail.**

**Step 3: Implement** — add to `adapter.py`:

```python
from dspy_qwen35_adapter.prompts import build_system_prompt


class Qwen35Adapter(Adapter):
    # ... existing code ...

    def format_field_description(self, signature: type[Signature]) -> str:
        return signature.instructions or ""

    def format_field_structure(self, signature: type[Signature]) -> str:
        return (
            "When calling a tool, emit exactly one <function=NAME>"
            "<parameter=KEY>VALUE</parameter>...</function>. "
            "Otherwise respond in plain text."
        )

    def format_task_description(self, signature: type[Signature]) -> str:
        return signature.instructions or ""

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        parts = []
        for name, _field in signature.input_fields.items():
            if name == "tools":
                continue
            value = inputs.get(name, "")
            parts.append(f"{name}: {value}")
        return (prefix + "\n\n".join(parts) + suffix).strip()

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        has_react_fields = REACT_TOOL_FIELDS.issubset(signature.output_fields.keys())
        if has_react_fields:
            thought = outputs.get("next_thought", "")
            name = outputs.get("next_tool_name", "")
            args = outputs.get("next_tool_args", {}) or {}
            params = "".join(
                f"<parameter={k}>{v}</parameter>" for k, v in args.items()
            )
            return f"{thought}\n<function={name}>{params}</function>".strip()
        return "\n".join(f"{k}: {v}" for k, v in outputs.items())

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        tools = inputs.get("tools") or []
        tool_list = tools if isinstance(tools, list) else [tools]
        system = build_system_prompt(
            task_description=signature.instructions or "",
            tools=tool_list,
        )
        user = self.format_user_message_content(signature, inputs)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
```

**Step 4: Run — expect pass.**

**Step 5: Commit**

```bash
git commit -am "feat: Qwen35Adapter format methods (system prompt + XML replay)"
```

---

## Task 10: Strict-parse behavior and non-ReAct signature fallback (TDD)

**Files:**
- Modify: `tests/test_adapter_integration.py`
- Modify: `dspy_qwen35_adapter/adapter.py`

**Step 1: Failing tests**

```python
def test_strict_parse_raises_when_no_call():
    from dspy.utils.exceptions import AdapterParseError
    a = Qwen35Adapter(strict_parse=True)
    sig = _sample_signature_with_tools()
    with pytest.raises(AdapterParseError):
        a.parse(sig, "Just some plain text")


def test_non_strict_returns_finish_when_no_call():
    a = Qwen35Adapter(strict_parse=False)
    sig = _sample_signature_with_tools()
    out = a.parse(sig, "Plain answer")
    assert out["next_tool_name"] == "finish"
    assert out["next_thought"] == "Plain answer"
    assert out["next_tool_args"] == {}
```

**Step 2: Run — should already pass from Task 8 (this task exists to lock the behavior with dedicated tests). If red, fix.**

**Step 3: Commit**

```bash
git commit -am "test: lock strict_parse and graceful-finish behaviors"
```

---

## Task 11: End-to-end ReAct mocked test

**Files:**
- Modify: `tests/test_adapter_integration.py`

**Step 1: Write the failing test**

```python
def test_react_completes_with_qwen_adapter_and_dummy_lm():
    """Run a full ReAct loop against DummyLM using Qwen35Adapter.
    Verifies format+parse round-trip without any network."""
    from dspy.utils.dummies import DummyLM

    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Weather in {city}: sunny, 72F."

    lm = DummyLM(
        [
            # Turn 1: call get_weather
            {"response": (
                "I'll check Tokyo's weather.\n"
                "<function=get_weather><parameter=city>Tokyo</parameter></function>"
            )},
            # Turn 2: finish
            {"response": "I have the answer.\n<function=finish></function>"},
            # Extract turn: final answer as plain text
            {"response": "Tokyo is sunny and 72F."},
        ]
    )

    with dspy.context(lm=lm, adapter=Qwen35Adapter()):
        react = dspy.ReAct("question -> answer", tools=[get_weather])
        pred = react(question="What's the weather in Tokyo?")

    assert pred.trajectory["tool_name_0"] == "get_weather"
    assert pred.trajectory["tool_args_0"] == {"city": "Tokyo"}
    assert "sunny" in pred.trajectory["observation_0"]
    assert pred.trajectory["tool_name_1"] == "finish"
```

**Step 2: Run — likely fails. Read the error carefully and fix minimally.**

Most likely issues: `DummyLM` response shape mismatch with our `format()` output, or the extract pass's `ChainOfThought` not finding the right fields. Adjust `adapter.py` or the test until it passes, but **do not** weaken the parser — the parser is locked by Tasks 2-6.

**Step 3: Run until green.**

**Step 4: Commit**

```bash
git commit -am "test: end-to-end ReAct with Qwen35Adapter against DummyLM"
```

---

## Task 12: Harness scenarios module

**Files:**
- Create: `harness/scenarios.py`

**Step 1: Write scenarios**

```python
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
```

**Step 2: Commit**

```bash
git add harness/scenarios.py
git commit -m "feat(harness): define 1/3/10-tool ReAct scenarios"
```

---

## Task 13: Harness runner with CSV output and trace capture

**Files:**
- Create: `harness/run_eval.py`

**Step 1: Write the runner**

```python
# harness/run_eval.py
"""Run a ReAct scenario N times against a chosen adapter, record per-run
metrics to CSV, and optionally dump raw LM content to harness/traces/.

Usage:
  python -m harness.run_eval --adapter qwen35 --scenario s3 --runs 20
  python -m harness.run_eval --adapter chat --scenario s10 --runs 20 --capture-traces
"""
import argparse
import csv
import datetime as dt
import logging
import os
import sys
import traceback
from pathlib import Path

import dspy
from dspy_qwen35_adapter import Qwen35Adapter
from harness.scenarios import ALL_SCENARIOS

RESULTS_DIR = Path(__file__).parent / "results"
TRACES_DIR = Path(__file__).parent / "traces"

LM_STUDIO_BASE = os.environ.get("LMSTUDIO_BASE", "http://localhost:1234/v1")
LM_MODEL = os.environ.get("QWEN_MODEL", "openai/qwen3.5-35b-a3b")


def build_adapter(name: str):
    if name == "qwen35":
        return Qwen35Adapter()
    if name == "chat":
        return dspy.ChatAdapter()
    if name == "json":
        return dspy.JSONAdapter()
    raise ValueError(name)


def run_once(scenario, adapter, capture_file: Path | None):
    react = dspy.ReAct("question -> answer", tools=scenario.tools)

    parse_failures = 0
    tool_exec_failures = 0

    # Monkey-patch adapter.parse to count failures without changing behavior.
    original_parse = adapter.parse
    def counting_parse(sig, completion):
        nonlocal parse_failures
        try:
            return original_parse(sig, completion)
        except Exception:
            parse_failures += 1
            raise
    adapter.parse = counting_parse

    try:
        pred = react(question=scenario.question)
        trajectory = pred.trajectory
        turns = sum(1 for k in trajectory if k.startswith("tool_name_"))
        answer = str(getattr(pred, "answer", "")).lower()
        task_succeeded = scenario.golden_answer_substring.lower() in answer
        tool_exec_failures = sum(
            1 for k, v in trajectory.items()
            if k.startswith("observation_") and "Execution error" in str(v)
        )
        max_iters_hit = turns >= react.max_iters
    except Exception as e:
        return {
            "turns_completed": 0,
            "max_iters_hit": False,
            "parse_failures": parse_failures,
            "tool_exec_failures": tool_exec_failures,
            "task_succeeded": False,
            "error": type(e).__name__,
        }
    finally:
        adapter.parse = original_parse

    return {
        "turns_completed": turns,
        "max_iters_hit": max_iters_hit,
        "parse_failures": parse_failures,
        "tool_exec_failures": tool_exec_failures,
        "task_succeeded": task_succeeded,
        "error": "",
    }


def maybe_install_trace_capture(capture_dir: Path, adapter_name: str, scenario_name: str, run_idx: int):
    """Wrap dspy.LM.__call__ to dump raw choices content to disk for later
    promotion into the fixture corpus."""
    if not capture_dir:
        return lambda: None
    original = dspy.LM.__call__
    turn = {"n": 0}
    def wrapped(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        for item in (out if isinstance(out, list) else [out]):
            text = item["text"] if isinstance(item, dict) else str(item)
            path = capture_dir / f"{scenario_name}-{adapter_name}-r{run_idx:03d}-t{turn['n']:02d}.txt"
            path.write_text(text)
            turn["n"] += 1
        return out
    dspy.LM.__call__ = wrapped
    return lambda: setattr(dspy.LM, "__call__", original)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["chat", "json", "qwen35"], required=True)
    ap.add_argument("--scenario", choices=list(ALL_SCENARIOS), required=True)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--capture-traces", action="store_true")
    args = ap.parse_args()

    scenario = ALL_SCENARIOS[args.scenario]
    adapter = build_adapter(args.adapter)
    lm = dspy.LM(LM_MODEL, api_base=LM_STUDIO_BASE, api_key="lm-studio")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = RESULTS_DIR / f"{stamp}-{args.adapter}-{args.scenario}.csv"

    with dspy.context(lm=lm, adapter=adapter):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "scenario", "adapter", "run_idx", "turns_completed",
                "max_iters_hit", "parse_failures", "tool_exec_failures",
                "task_succeeded", "error",
            ])
            for i in range(args.runs):
                trace_dir = TRACES_DIR if args.capture_traces else None
                restore = maybe_install_trace_capture(trace_dir, args.adapter, args.scenario, i)
                try:
                    result = run_once(scenario, adapter, trace_dir)
                finally:
                    restore()
                w.writerow([
                    args.scenario, args.adapter, i, result["turns_completed"],
                    result["max_iters_hit"], result["parse_failures"],
                    result["tool_exec_failures"], result["task_succeeded"],
                    result["error"],
                ])
                print(f"run {i}: parse_fail={result['parse_failures']} "
                      f"tool_fail={result['tool_exec_failures']} "
                      f"ok={result['task_succeeded']}")

    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke-test with a trivial dry run**

Run: `python -m harness.run_eval --adapter qwen35 --scenario s1 --runs 1`

This requires LM Studio running locally. If LM Studio isn't available, skip and mark as "needs LM runtime — see Task 15." Do not commit if smoke-test fails because of code bugs; do commit if it fails only because LM Studio isn't running.

**Step 3: Commit**

```bash
git add harness/run_eval.py
git commit -m "feat(harness): ReAct runner with CSV output and trace capture"
```

---

## Task 14: Analysis script

**Files:**
- Create: `harness/analyze.py`

**Step 1: Write the summarizer**

```python
# harness/analyze.py
"""Summarize all CSVs under harness/results/ as a markdown table comparing
adapters across scenarios on parse-failure rate and task success rate."""
import csv
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    cells = defaultdict(list)
    for path in sorted(RESULTS_DIR.glob("*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["scenario"], row["adapter"])
                cells[key].append(row)

    scenarios = sorted({k[0] for k in cells})
    adapters = sorted({k[1] for k in cells})

    print("| scenario | adapter | runs | parse_fail / run | tool_fail / run | task_success |")
    print("|---|---|---|---|---|---|")
    for s in scenarios:
        for a in adapters:
            rows = cells.get((s, a), [])
            if not rows:
                continue
            n = len(rows)
            pf = sum(int(r["parse_failures"]) for r in rows) / n
            tf = sum(int(r["tool_exec_failures"]) for r in rows) / n
            ok = sum(r["task_succeeded"] == "True" for r in rows) / n
            print(f"| {s} | {a} | {n} | {pf:.2f} | {tf:.2f} | {ok:.0%} |")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke-test**

Run: `python -m harness.analyze`
Expected: either an empty table (no CSVs yet) or a populated one.

**Step 3: Commit**

```bash
git add harness/analyze.py
git commit -m "feat(harness): CSV -> markdown comparison table"
```

---

## Task 15: First live benchmark run

**Prerequisites:** LM Studio running with Qwen 3.5 35B A3B (Unsloth GGUF) on `http://localhost:1234`.

**Step 1: Run all nine cells**

```bash
for ADAPTER in chat json qwen35; do
  for SCENARIO in s1 s3 s10; do
    python -m harness.run_eval --adapter $ADAPTER --scenario $SCENARIO --runs 20 --capture-traces
  done
done
```

Expect this to take 30-90 minutes depending on model speed. If any cell crashes hard (unhandled exception), fix the bug in the corresponding adapter/harness file and rerun **only that cell**.

**Step 2: Analyze**

```bash
python -m harness.analyze > harness/results/summary.md
```

**Step 3: Check success criteria**

Open `harness/results/summary.md`. Verify:
- `qwen35` has ≤ 1/3 the `parse_fail / run` of `chat` on `s3` and `s10`.
- `qwen35` `task_success` ≥ `chat` and `json` on all three.

**If criteria not met:** go to Task 16. If met: commit results and skip to Task 17.

**Step 4: Commit results**

```bash
git add harness/results/summary.md harness/results/*.csv
git commit -m "bench: first live benchmark results"
```

---

## Task 16: Iterate — promote real failures into fixtures, tune parser

**Only run if Task 15 did not meet success criteria.**

**Step 1: Find parse-fail traces**

```bash
ls harness/traces/ | head -30
# For any trace from a qwen35 run where parse_failures > 0 or task_succeeded was False,
# inspect the file and diagnose.
```

**Step 2: For each novel failure mode:**

1. Copy the trace file into `tests/fixtures/traces/NN-<label>.txt`.
2. Write the matching `.json` describing the correct expected parse.
3. Run `pytest tests/test_parsing.py -v`. It should fail for the new fixture.
4. Adjust `dspy_qwen35_adapter/parsing.py` minimally to handle the failure mode **without breaking existing fixtures**.
5. Run `pytest tests/ -v`. All should pass.
6. Commit: `git commit -am "fix(parsing): handle <failure mode> (fixture NN)"`

**Step 3: Re-run benchmark on the failing scenario only**

```bash
python -m harness.run_eval --adapter qwen35 --scenario s10 --runs 20 --capture-traces
python -m harness.analyze > harness/results/summary.md
```

Repeat Task 16 until criteria are met. **Do not relax the criteria.** If the parser can't meet them after three iteration cycles, stop and escalate — the problem may be the prompt template, not the parser.

**Step 4: Commit when green**

```bash
git commit -am "bench: success criteria met after parser tuning"
```

---

## Task 17: README and PyPI prep

**Files:**
- Create: `README.md`
- Modify: `pyproject.toml` (add URLs, classifiers)

**Step 1: Write `README.md`**

Focus on the quickstart and a one-line explanation of why this exists. Link to the design doc for the full motivation. Reference the latest benchmark summary. Keep under 200 lines.

Include sections:
- What it is (2 sentences)
- Install (`pip install dspy-qwen35-adapter`)
- Quickstart (the canonical 5-line snippet)
- Why (link to design doc + Reddit notes)
- Benchmarks (paste the summary table)
- License

**Step 2: Add classifiers and URLs to `pyproject.toml`**

```toml
[project.urls]
Homepage = "https://github.com/<user>/dspy-qwen35-adapter"
Issues = "https://github.com/<user>/dspy-qwen35-adapter/issues"

[project.classifiers]
# (use a list, not a table — this is a placeholder; pick from pypi.org/classifiers)
```

Actual `classifiers` goes in `[project]` as a list. Fill in:

```toml
classifiers = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
```

**Step 3: Build and inspect the sdist**

```bash
pip install build
python -m build
ls dist/
tar -tzf dist/dspy_qwen35_adapter-0.1.0.tar.gz | head -30
```

Verify: `dspy/`, `harness/`, `tests/`, `docs/` are NOT in the tarball; `dspy_qwen35_adapter/` IS.

**Step 4: Commit**

```bash
git add README.md pyproject.toml
git commit -m "docs: README with quickstart and benchmark results"
```

---

## Task 18: Tag v0.1.0 (optional — only when ready to publish)

Do not run this task automatically. Stop here and confirm with the user that they want to publish.

```bash
git tag v0.1.0
git push origin main --tags
# python -m twine upload dist/*   # only after user confirms
```

---

## Success definition

Plan is complete when:
- `pytest tests/ -v` is all green.
- `harness/results/summary.md` shows `qwen35` beating `chat` on parse-failure rate by at least 3× on S3 and S10.
- `python -m build` produces a clean sdist with correct file inclusion.
- User has a working quickstart that matches the design doc's DX promise.
