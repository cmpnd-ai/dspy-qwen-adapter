import ast
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

# Fallback: DSPy-style XML field tags the model emits when it drifts from
# Qwen-native format (e.g. <next_tool_name>, <next_tool_args>).
_DSPY_REACT_XML = re.compile(
    r"<(next_thought|next_tool_name|next_tool_args)>([\s\S]*?)</\1>",
    re.IGNORECASE,
)


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
    first <function=...> tag; tool_call is the first parsed function block.

    Falls back to DSPy-style XML field tags (<next_tool_name>, <next_tool_args>)
    when the model drifts from Qwen-native format."""
    fn_match = _FUNCTION_BLOCK.search(text)
    if fn_match:
        thought = text[: fn_match.start()].strip()
        name = fn_match.group(1)
        args: dict[str, Any] = {}
        for p in _PARAMETER_BLOCK.finditer(fn_match.group(2)):
            args[p.group(1)] = _decode_value(p.group(2))
        return thought, (name, args)

    # Fallback: model used DSPy-style XML field tags instead of Qwen-native format.
    fields: dict[str, str] = {
        m.group(1).lower(): m.group(2).strip()
        for m in _DSPY_REACT_XML.finditer(text)
    }
    if "next_tool_name" in fields:
        thought = fields.get("next_thought", "")
        name = fields["next_tool_name"]
        args_raw = fields.get("next_tool_args", "")
        args = {}
        if args_raw:
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(args_raw)
                    if isinstance(parsed, dict):
                        args = parsed
                except (ValueError, SyntaxError):
                    pass
        return thought, (name, args)

    return text.strip(), None


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


# Qwen 3 (pre-3.5) Hermes-style calls: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
_HERMES_BLOCK = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>")


def _strip_param(raw: str) -> str:
    """Drop the single newline the Qwen 3.5 template puts around values."""
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    return raw


def decode_for_schema(raw: str, schema: dict[str, Any] | None) -> Any:
    """Decode a raw <parameter> value the way the Qwen 3.5 chat template
    encodes it: strings are written verbatim, everything else as JSON."""
    types = {schema.get("type")} if schema and "type" in schema else {
        s.get("type") for s in (schema or {}).get("anyOf", [])
    }
    if "string" in types and types <= {"string", "null"}:
        return _strip_param(raw).strip()
    return _decode_value(raw)


def split_thought_and_calls(
    text: str,
) -> tuple[str, list[tuple[str, dict[str, str]]]]:
    """Split content into (thought, calls). Thought is the text before the
    first call. Each call is (name, raw_args) with parameter values left as
    undecoded strings so the caller can decode them against a tool schema.

    Recognizes Qwen 3.5 `<function=NAME>` blocks first, then Qwen 3
    Hermes-style JSON `<tool_call>` blocks."""
    fn_matches = list(_FUNCTION_BLOCK.finditer(text))
    if fn_matches:
        first = fn_matches[0].start()
        # Include an opening <tool_call> wrapper in the cut, if present.
        opener = text.rfind("<tool_call>", 0, first)
        cut = opener if opener != -1 and text[opener:first].strip() == "<tool_call>" else first
        calls = [
            (m.group(1), {p.group(1): p.group(2) for p in _PARAMETER_BLOCK.finditer(m.group(2))})
            for m in fn_matches
        ]
        return text[:cut].strip(), calls

    calls = []
    hermes_matches = list(_HERMES_BLOCK.finditer(text))
    for m in hermes_matches:
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "name" not in payload:
            continue
        args = payload.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        # Hermes args are already JSON-typed; re-serialize non-strings so
        # decode_for_schema treats both formats the same way.
        raw = {k: v if isinstance(v, str) else json.dumps(v) for k, v in args.items()}
        calls.append((payload["name"], raw))
    if calls:
        return text[: hermes_matches[0].start()].strip(), calls

    return text.strip(), []
