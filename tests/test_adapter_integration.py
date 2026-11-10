import dspy
import pytest
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.lm15 import Message, Response, TextPart, ThinkingPart, Usage
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import AdapterParseError

from dspy_qwen_adapter import QwenAdapter
from dspy_qwen_adapter.adapter import _ACTIVE_TOOLS


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Weather in {city}: sunny, 72F."


def _react_signature():
    """The inner signature `dspy.ReAct` builds for `question -> answer`."""
    return dspy.ReAct("question -> answer", tools=[get_weather]).react.signature


def _react_inputs(question="What's the weather in Tokyo?"):
    react = dspy.ReAct("question -> answer", tools=[get_weather])
    return {
        "question": question,
        "history": dspy.History(messages=[]),
        "tools": list(react.tools.values()),
        "turn_budget": "Turn 1 of 20 (19 more after this one).",
    }


def _response(*parts):
    return Response(
        id="r1",
        model="m",
        message=Message(role="assistant", parts=parts),
        finish_reason="stop",
        usage=Usage(),
    )


# -------- construction --------


def test_adapter_instantiates():
    a = QwenAdapter()
    assert a.use_native_function_calling is False
    assert a.strict_parse is False


def test_adapter_accepts_strict_parse_flag():
    assert QwenAdapter(strict_parse=True).strict_parse is True


# -------- parse --------


def test_parse_returns_thought_and_tool_calls():
    out = QwenAdapter().parse(
        _react_signature(),
        "I should check.\n<function=get_weather><parameter=city>Tokyo</parameter></function>",
    )
    assert out["next_thought"] == "I should check."
    assert isinstance(out["tool_calls"], ToolCalls)
    call = out["tool_calls"].tool_calls[0]
    assert (call.name, call.args) == ("get_weather", {"city": "Tokyo"})


def test_parse_strips_think_before_splitting():
    out = QwenAdapter().parse(
        _react_signature(),
        "<think>internal</think>I should check.\n"
        "<function=get_weather><parameter=city>Tokyo</parameter></function>",
    )
    assert out["next_thought"] == "I should check."


def test_parse_excludes_tool_call_wrapper_from_thought():
    out = QwenAdapter().parse(
        _react_signature(),
        "Looking.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>",
    )
    assert out["next_thought"] == "Looking."
    assert out["tool_calls"].tool_calls[0].args == {"city": "Oslo"}


def test_parse_collects_multiple_calls():
    out = QwenAdapter().parse(
        _react_signature(),
        "Both.\n<function=get_weather><parameter=city>A</parameter></function>"
        "<function=get_weather><parameter=city>B</parameter></function>",
    )
    assert [c.args["city"] for c in out["tool_calls"].tool_calls] == ["A", "B"]


def test_parse_decodes_args_against_tool_schema():
    """String-typed params stay strings even when they look like JSON;
    other types are JSON-decoded, matching the Qwen 3.5 chat template."""

    def lookup(code: str, limit: int) -> str:
        """Look up a code."""
        return ""

    token = _ACTIVE_TOOLS.set({"lookup": Tool(lookup)})
    try:
        out = QwenAdapter().parse(
            _react_signature(),
            "<function=lookup><parameter=code>\n007\n</parameter>"
            "<parameter=limit>\n5\n</parameter></function>",
        )
    finally:
        _ACTIVE_TOOLS.reset(token)
    assert out["tool_calls"].tool_calls[0].args == {"code": "007", "limit": 5}


def test_parse_accepts_hermes_json_tool_call():
    """Qwen 3 (pre-3.5) models emit JSON inside <tool_call>."""
    out = QwenAdapter().parse(
        _react_signature(),
        'Checking.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Lima"}}\n</tool_call>',
    )
    assert out["next_thought"] == "Checking."
    call = out["tool_calls"].tool_calls[0]
    assert (call.name, call.args) == ("get_weather", {"city": "Lima"})


def test_strict_parse_raises_when_no_call():
    with pytest.raises(AdapterParseError):
        QwenAdapter(strict_parse=True).parse(_react_signature(), "Just some plain text")


def test_non_strict_returns_empty_tool_calls_when_no_call():
    out = QwenAdapter().parse(_react_signature(), "Plain answer")
    assert out["tool_calls"].tool_calls == []
    assert out["next_thought"] == "Plain answer"


def test_parse_non_tool_signature_delegates_to_xml_adapter():
    class Sig(dspy.Signature):
        """Answer the question."""
        question: str = dspy.InputField()
        reasoning: str = dspy.OutputField()
        answer: str = dspy.OutputField()

    out = QwenAdapter().parse(
        Sig,
        "<think>let me think</think>\n"
        "<reasoning>\nTokyo has mild spring weather.\n</reasoning>\n"
        "<answer>\nTokyo is sunny and 72F.\n</answer>",
    )
    assert out["reasoning"] == "Tokyo has mild spring weather."
    assert out["answer"] == "Tokyo is sunny and 72F."


# -------- format --------


def test_format_tool_signature_builds_qwen_system_prompt():
    prompt = QwenAdapter().format(_react_signature(), [], _react_inputs())
    assert "<tools>" in prompt.system
    assert "get_weather" in prompt.system
    assert "<function=" in prompt.system
    assert "`submit`" in prompt.system


def test_format_user_message_omits_tools_history_and_footer():
    prompt = QwenAdapter().format(_react_signature(), [], _react_inputs())
    assert len(prompt.messages) == 1
    user = prompt.messages[0].parts[0].text
    assert "<question>\nWhat's the weather in Tokyo?\n</question>" in user
    assert "<turn_budget>" in user
    assert "<tools>" not in user
    assert "Respond with" not in user


def test_format_assistant_tool_call_is_canonical_qwen_35():
    content = QwenAdapter().format_assistant_message_content(
        _react_signature(),
        outputs={
            "next_thought": "",
            "tool_calls": ToolCalls.from_dict_list([{"name": "search", "args": {"query": "weather", "limit": 5}}]),
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


def test_format_assistant_includes_thought_before_call():
    content = QwenAdapter().format_assistant_message_content(
        _react_signature(),
        outputs={
            "next_thought": "check weather",
            "tool_calls": ToolCalls.from_dict_list([{"name": "get_weather", "args": {"city": "Tokyo"}}]),
        },
    )
    assert content.startswith("check weather\n<tool_call>")
    assert "<parameter=city>\nTokyo\n</parameter>" in content


def test_format_replays_history_as_qwen_transcript():
    """A past ReAct turn renders as assistant <tool_call> XML followed by a
    single user turn holding the <tool_response> and the next turn budget."""
    calls = ToolCalls.from_dict_list([{"id": "c0", "name": "get_weather", "args": {"city": "Tokyo"}}])
    results = ToolCallResults.from_tool_calls_and_values(calls, ["sunny, 72F"], [False])
    history = dspy.History(messages=[{
        "question": "What's the weather in Tokyo?",
        "turn_budget": "Turn 1 of 20 (19 more after this one).",
        "next_thought": "I should check.",
        "tool_calls": calls.model_copy(update={"tool_call_results": results}),
    }])
    inputs = {**_react_inputs(), "history": history, "turn_budget": "Turn 2 of 20 (18 more after this one)."}
    del inputs["question"]

    prompt = QwenAdapter().format(_react_signature(), [], inputs)
    roles = [m.role for m in prompt.messages]
    assert roles == ["user", "assistant", "user"]

    assistant = prompt.messages[1].parts[0].text
    assert assistant.startswith("I should check.\n<tool_call>\n<function=get_weather>")

    last_user = prompt.messages[2].parts[0].text
    assert '<tool_response name="get_weather">\nsunny, 72F\n</tool_response>' in last_user
    assert "Turn 2 of 20" in last_user


def test_format_demos_use_qwen_xml():
    demo = {
        "question": "What is the weather in Paris?",
        "turn_budget": "Turn 1 of 20 (19 more after this one).",
        "next_thought": "Checking Paris.",
        "tool_calls": ToolCalls.from_dict_list([{"name": "get_weather", "args": {"city": "Paris"}}]),
    }
    prompt = QwenAdapter().format(_react_signature(), [demo], _react_inputs())
    assistants = [m for m in prompt.messages if m.role == "assistant"]
    assert len(assistants) == 1
    content = assistants[0].parts[0].text
    assert "<function=get_weather>" in content
    assert "<tool_calls>" not in content
    assert "Paris" in prompt.messages[0].parts[0].text


# -------- end-to-end --------


def test_react_completes_with_qwen_adapter_and_dummy_lm():
    lm = DummyLM([
        {"response": "I'll check Tokyo.\n<function=get_weather><parameter=city>Tokyo</parameter></function>"},
        {"response": "Done.\n<function=submit><parameter=answer>72</parameter></function>"},
    ])
    with dspy.context(lm=lm, adapter=QwenAdapter()):
        pred = dspy.ReAct("question -> answer", tools=[get_weather])(question="What's the weather in Tokyo?")

    assert pred.termination_reason == "submit"
    # `answer` is a string field, so "72" must not be JSON-decoded to an int.
    assert pred.answer == "72"
    first = pred.history.messages[0]["tool_calls"]
    assert first.tool_calls[0].args == {"city": "Tokyo"}
    assert "sunny" in str(first.tool_call_results)


def test_react_forces_submit_after_plain_text_answer():
    lm = DummyLM([
        {"response": "It is sunny."},
        {"response": "<function=submit><parameter=answer>sunny</parameter></function>"},
    ])
    with dspy.context(lm=lm, adapter=QwenAdapter()):
        pred = dspy.ReAct("question -> answer", tools=[get_weather])(question="Weather?")
    assert pred.termination_reason == "forced_submit"
    assert pred.answer == "sunny"


# -------- reasoning rescue --------


class _AnswerSig(dspy.Signature):
    """Answer."""
    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    answer: str = dspy.OutputField()


def test_postprocess_promotes_thinking_when_text_empty():
    """LM Studio's reasoning parser can route the whole completion into the
    thinking channel. The adapter promotes it so parse() has input."""
    response = _response(ThinkingPart(
        "The user asked about Tokyo.\n"
        "<reasoning>\nLookup returned sunny.\n</reasoning>\n"
        "<answer>\nTokyo is sunny.\n</answer>"
    ))
    values = QwenAdapter()._call_postprocess(_AnswerSig, _AnswerSig, [response])
    assert values[0]["answer"] == "Tokyo is sunny."


def test_postprocess_passthrough_when_text_present():
    response = _response(
        ThinkingPart("Hidden thinking that must NOT leak."),
        TextPart("<reasoning>\nVisible.\n</reasoning>\n<answer>\nVisible answer.\n</answer>"),
    )
    values = QwenAdapter()._call_postprocess(_AnswerSig, _AnswerSig, [response])
    assert values[0]["answer"] == "Visible answer."


def test_postprocess_appends_tool_call_found_only_in_thinking():
    sig = _react_signature()
    response = _response(
        ThinkingPart("<function=get_weather><parameter=city>Rome</parameter></function>"),
        TextPart("Let me check Rome."),
    )
    values = QwenAdapter()._call_postprocess(sig, sig, [response])
    assert values[0]["tool_calls"].tool_calls[0].args == {"city": "Rome"}


def test_postprocess_raises_when_response_empty():
    """DSPy 3.5 treats an empty completion as a parse error."""
    with pytest.raises(AdapterParseError):
        QwenAdapter()._call_postprocess(_AnswerSig, _AnswerSig, [_response(TextPart(""))])


def test_postprocess_does_not_promote_deliberation():
    """Thinking that only talks about the output tags (e.g. a model that ran
    out of tokens mid-deliberation) must fail to parse, not yield a field
    scraped from the reasoning."""
    response = _response(ThinkingPart(
        "The format says <answer>\n{answer}\n</answer>. Should I replace it? "
        "Wait, maybe it means"
    ))
    with pytest.raises(AdapterParseError):
        QwenAdapter()._call_postprocess(_AnswerSig, _AnswerSig, [response])


def test_postprocess_reads_reasoning_text_from_responses_api_output():
    """Over the Responses API, LM Studio puts Qwen's reasoning in
    output[].content[].reasoning_text, which lm15 leaves out of ThinkingPart."""
    import dataclasses

    response = dataclasses.replace(
        _response(ThinkingPart("")),
        provider_data={"output": [{
            "type": "reasoning",
            "summary": [],
            "content": [{
                "type": "reasoning_text",
                "text": "<reasoning>\nok\n</reasoning>\n<answer>\nParis\n</answer>",
            }],
        }]},
    )
    values = QwenAdapter()._call_postprocess(_AnswerSig, _AnswerSig, [response])
    assert values[0]["answer"] == "Paris"


def test_react_async_path_decodes_args_with_tool_schema():
    import asyncio

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    lm = DummyLM([
        {"response": "<function=add><parameter=a>2</parameter><parameter=b>3</parameter></function>"},
        {"response": "<function=submit><parameter=answer>5</parameter></function>"},
    ])

    async def run():
        with dspy.context(lm=lm, adapter=QwenAdapter()):
            return await dspy.ReAct("question -> answer", tools=[add]).acall(question="2+3?")

    pred = asyncio.run(run())
    assert pred.answer == "5"
    first = pred.history.messages[0]["tool_calls"]
    assert first.tool_calls[0].args == {"a": 2, "b": 3}
    assert first.tool_call_results.tool_call_results[0].value == 5


def test_forced_submit_shows_only_submit_tool():
    """ReAct forces `submit` with tool_choice when it runs out of turns. The
    adapter enforces that in the prompt, since it sends no tools natively."""
    lm = DummyLM([
        {"response": "<function=get_weather><parameter=city>Tokyo</parameter></function>"},
        {"response": "<function=submit><parameter=answer>sunny</parameter></function>"},
    ])
    with dspy.context(lm=lm, adapter=QwenAdapter()):
        pred = dspy.ReAct("question -> answer", tools=[get_weather], max_iters=1)(question="Weather?")
    assert pred.termination_reason == "forced_submit"

    first_system = lm.history[0]["messages"][0]["content"]
    forced_system = lm.history[1]["messages"][0]["content"]
    assert '<tool name="get_weather">' in first_system
    assert '<tool name="get_weather">' not in forced_system
    assert '<tool name="submit">' in forced_system
    assert "you must call `submit`" in forced_system
