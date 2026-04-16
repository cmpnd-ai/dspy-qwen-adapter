import json
from xml.sax.saxutils import escape, quoteattr
from typing import Iterable
from dspy.adapters.types.tool import Tool


_EXEMPLAR = (
    "Example of a tool call:\n"
    "<function=example_tool>"
    "<parameter=arg1>value1</parameter>"
    "<parameter=arg2>value2</parameter>"
    "</function>"
)


_REACT_GUIDANCE = (
    "When the required outputs are next_thought, next_tool_name, and "
    "next_tool_args, do NOT emit JSON. Instead write your reasoning as "
    "plain text (that becomes next_thought), then emit a single tool call "
    "on the following lines:\n"
    "<function=TOOL_NAME><parameter=KEY>VALUE</parameter>...</function>\n"
    "The tag name is next_tool_name; the parameters inside form "
    "next_tool_args. Example:\n"
    "I need to check Tokyo's weather.\n"
    "<function=get_weather><parameter=city>Tokyo</parameter></function>"
)


def _tool_to_xml(tool: Tool) -> str:
    schema = tool.args or {}
    return (
        f"  <tool name={quoteattr(tool.name)}>\n"
        f"    <description>{escape(tool.desc or '')}</description>\n"
        f"    <parameters>{json.dumps(schema)}</parameters>\n"
        f"  </tool>"
    )


def build_system_prompt(
    task_description: str,
    tools: Iterable[Tool],
    react_fields: bool = False,
) -> str:
    """Build a system prompt that keeps Qwen 3.5 in-distribution for tool use.
    Emits a <tools> block and a one-shot XML exemplar only when tools are
    provided. When react_fields=True, uses ReAct-specific guidance that maps
    next_thought / next_tool_name / next_tool_args to Qwen XML."""
    tool_list = list(tools)
    sections = [task_description.strip()] if task_description else []
    if tool_list:
        block = "<tools>\n" + "\n".join(_tool_to_xml(t) for t in tool_list) + "\n</tools>"
        sections.append(block)
        if react_fields:
            sections.append(_REACT_GUIDANCE)
        else:
            sections.append(_EXEMPLAR)
            sections.append(
                "Emit exactly one <function=...>...</function> per turn, or answer in plain text when done."
            )
    return "\n\n".join(sections)
