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


def test_parse_non_react_populates_last_output_field():
    """Non-ReAct signatures (e.g., ChainOfThought extract) should put cleaned
    content into the last output field — typically the user's declared answer,
    not the ChainOfThought-prepended reasoning."""
    class Sig(dspy.Signature):
        """Answer the question."""
        question: str = dspy.InputField()
        reasoning: str = dspy.OutputField()
        answer: str = dspy.OutputField()

    a = Qwen35Adapter()
    out = a.parse(Sig, "Tokyo is sunny and 72F.")
    assert out["answer"] == "Tokyo is sunny and 72F."
    assert out["reasoning"] == ""


def test_format_react_signature_emits_xml_protocol_and_drops_json_directive():
    """In ReAct mode, format() must (a) emit Qwen XML guidance and (b) NOT
    leak ReAct's auto-generated 'next_tool_args must be in JSON format' line
    from signature.instructions — that line is what causes Qwen to emit JSON."""
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
    system = next(m for m in messages if m["role"] == "system")["content"]
    assert "<function=" in system
    assert "<parameter=" in system
    # The JSON directive from signature.instructions must NOT leak through.
    assert "JSON" not in system and "json" not in system
