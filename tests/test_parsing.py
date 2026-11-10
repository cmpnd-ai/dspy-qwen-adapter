import json
from pathlib import Path

import pytest

from dspy_qwen_adapter.parsing import (
    coerce_args_to_schema,
    extract_tool_call,
    split_thought_and_call,
    strip_think,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "traces"


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


def _fixture_pairs():
    txts = {p.stem for p in FIXTURE_DIR.glob("*.txt")}
    jsons = {p.stem for p in FIXTURE_DIR.glob("*.json")}
    orphans = txts.symmetric_difference(jsons)
    if orphans:
        raise RuntimeError(
            f"Orphan fixture files (missing .txt or .json mate): {sorted(orphans)}"
        )
    for stem in sorted(txts):
        yield pytest.param(
            FIXTURE_DIR / f"{stem}.txt",
            FIXTURE_DIR / f"{stem}.json",
            id=stem,
        )


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


# -------- multi-call and schema-aware decoding --------

from dspy_qwen_adapter.parsing import decode_for_schema, split_thought_and_calls  # noqa: E402


def test_split_calls_returns_raw_values():
    thought, calls = split_thought_and_calls(
        "Go.\n<function=f><parameter=n>\n5\n</parameter></function>"
    )
    assert thought == "Go."
    assert calls == [("f", {"n": "\n5\n"})]


def test_split_calls_no_call():
    assert split_thought_and_calls("  just text ") == ("just text", [])


def test_split_calls_hermes_reserializes_non_strings():
    _, calls = split_thought_and_calls(
        '<tool_call>{"name": "f", "arguments": {"n": 5, "s": "x"}}</tool_call>'
    )
    assert calls == [("f", {"n": "5", "s": "x"})]


def test_split_calls_hermes_ignores_bad_json():
    assert split_thought_and_calls("<tool_call>{not json}</tool_call>")[1] == []


def test_decode_string_schema_keeps_text():
    assert decode_for_schema("\n42\n", {"type": "string"}) == "42"


def test_decode_optional_string_schema_keeps_text():
    schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert decode_for_schema("[1]", schema) == "[1]"


def test_decode_non_string_schema_parses_json():
    assert decode_for_schema("\n[1, 2]\n", {"type": "array"}) == [1, 2]


def test_decode_without_schema_falls_back_to_json():
    assert decode_for_schema("true", None) is True
