import json
from typing import Iterable
from dspy.adapters.types.tool import Tool


_EXEMPLAR = (
    "Example of a tool call:\n"
    "<function=example_tool>"
    "<parameter=arg1>value1</parameter>"
    "<parameter=arg2>value2</parameter>"
    "</function>"
)


def _tool_to_xml(tool: Tool) -> str:
    schema = tool.args or {}
    return (
        f'  <tool name="{tool.name}">\n'
        f'    <description>{tool.desc or ""}</description>\n'
        f'    <parameters>{json.dumps(schema)}</parameters>\n'
        f'  </tool>'
    )


def build_system_prompt(task_description: str, tools: Iterable[Tool]) -> str:
    """Build a system prompt that keeps Qwen 3.5 in-distribution for tool use.
    Emits a <tools> block and a one-shot XML exemplar only when tools are
    provided."""
    tool_list = list(tools)
    sections = [task_description.strip()] if task_description else []
    if tool_list:
        block = "<tools>\n" + "\n".join(_tool_to_xml(t) for t in tool_list) + "\n</tools>"
        sections.append(block)
        sections.append(_EXEMPLAR)
        sections.append(
            "Emit exactly one <function=...>...</function> per turn, or answer in plain text when done."
        )
    return "\n\n".join(sections)
