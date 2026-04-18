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
    assert "<tool_call>" in content
    assert "<function=get_weather>" in content
    assert "</tool_call>" in content
    # Canonical Qwen 3.5 format puts the parameter value on its own lines.
    assert "<parameter=city>\nTokyo\n</parameter>" in content
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


def test_format_assistant_tool_call_is_canonical_qwen_35():
    """The replayed assistant turn must match Qwen 3.5's trained chat_template
    format: <tool_call><function=NAME><parameter=K>\\nVALUE\\n</parameter>...
    </function></tool_call>."""
    a = Qwen35Adapter()
    sig = _sample_signature_with_tools()
    content = a.format_assistant_message_content(
        sig,
        outputs={
            "next_thought": "",
            "next_tool_name": "search",
            "next_tool_args": {"query": "weather", "limit": 5},
        },
    )
    assert content == (
        "<tool_call>\n"
        "<function=search>\n"
        "<parameter=query>\nweather\n</parameter>\n"
        "<parameter=limit>\n5\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


def test_format_trajectory_renders_qwen_native_transcript():
    """When format_user_message_content is called with a trajectory-shaped
    input dict (ReAct's replay path), the output must be a Qwen-native
    transcript: reasoning as prose, tool calls in canonical XML, observations
    wrapped in <tool_response>."""
    a = Qwen35Adapter()

    trajectory = {
        "thought_0": "I should check the weather.",
        "tool_name_0": "get_weather",
        "tool_args_0": {"city": "Tokyo"},
        "observation_0": "sunny, 72F",
        "thought_1": "Got it. Done.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }
    sig = dspy.Signature(f"{', '.join(trajectory)} -> x")
    rendered = a.format_user_message_content(sig, trajectory)

    assert "<tool_call>\n<function=get_weather>" in rendered
    assert "<parameter=city>\nTokyo\n</parameter>" in rendered
    # The tool_response tag carries a name= attribute naming the tool that
    # produced it — helps small models (4B) preserve tool-output provenance
    # across the extract turn.
    assert '<tool_response name="get_weather">\nsunny, 72F\n</tool_response>' in rendered
    assert "<tool_call>\n<function=finish>\n</function>\n</tool_call>" in rendered
    # Non-trajectory path is unchanged: plain "name: value" lines.
    class Plain(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
    assert a.format_user_message_content(Plain, {"question": "hi"}) == "question: hi"


def test_postprocess_promotes_reasoning_content_when_text_empty():
    """When LM Studio's reasoning parser routes everything into
    reasoning_content and leaves text empty, we must fall back to using
    reasoning_content so parse() runs instead of returning all-None."""
    a = Qwen35Adapter()

    class Sig(dspy.Signature):
        """Answer."""
        question: str = dspy.InputField()
        reasoning: str = dspy.OutputField()
        answer: str = dspy.OutputField()

    outputs = [{
        "text": "",
        "reasoning_content": (
            "The user asked about Tokyo weather. I should say it's sunny.\n"
            "Final answer: Tokyo is sunny."
        ),
    }]
    values = a._call_postprocess(Sig, Sig, outputs, lm=None, lm_kwargs={})
    assert len(values) == 1
    # The answer is routed to the LAST output field (Task-10 fallback contract).
    assert values[0]["answer"], "answer should be populated from reasoning_content"
    assert "Tokyo is sunny" in values[0]["answer"]


def test_postprocess_passthrough_when_text_present():
    """When text is non-empty, reasoning_content must be ignored — the
    server has already split reasoning into a side channel and the visible
    text is what the parser should consume."""
    a = Qwen35Adapter()

    class Sig(dspy.Signature):
        """Answer."""
        question: str = dspy.InputField()
        reasoning: str = dspy.OutputField()
        answer: str = dspy.OutputField()

    outputs = [{
        "text": "Visible answer text.",
        "reasoning_content": "Hidden thinking that must NOT leak into the answer.",
    }]
    values = a._call_postprocess(Sig, Sig, outputs, lm=None, lm_kwargs={})
    assert values[0]["answer"] == "Visible answer text."
    assert "Hidden thinking" not in values[0]["answer"]


def test_postprocess_noop_when_both_empty():
    """If text is empty AND reasoning_content is empty/missing, fall through
    to the base class (which assigns None to every output field)."""
    a = Qwen35Adapter()

    class Sig(dspy.Signature):
        """Answer."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    outputs = [{"text": ""}]
    values = a._call_postprocess(Sig, Sig, outputs, lm=None, lm_kwargs={})
    assert values[0]["answer"] is None
