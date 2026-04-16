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
