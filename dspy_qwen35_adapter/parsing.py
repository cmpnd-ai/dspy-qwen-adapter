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


import json
from typing import Any


_FUNCTION_BLOCK = re.compile(r"<function=(\S+?)>([\s\S]*?)</function>")
_PARAMETER_BLOCK = re.compile(r"<parameter=(\S+?)>([\s\S]*?)</parameter>")


def _decode_value(raw: str) -> Any:
    raw = raw.strip()
    if raw == "":
        return ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
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
