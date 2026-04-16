import json
import re
from typing import Any

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
