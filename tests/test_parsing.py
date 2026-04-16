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
