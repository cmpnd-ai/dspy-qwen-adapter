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
