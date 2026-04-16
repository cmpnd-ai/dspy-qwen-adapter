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
    "You are an agent. Use the supplied tools to answer the user's question. "
    "On each turn, write plain-text reasoning on one or more lines, then emit "
    "exactly one tool call in this form on its own line:\n"
    "  <function=TOOL_NAME><parameter=KEY>VALUE</parameter>...</function>\n"
    "Emit nothing after the closing </function>. When you have enough "
    "information to answer, call <function=finish></function>.\n"
    "Example of a correct turn:\n"
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
    sections: list[str] = []
    # In ReAct mode, lead with the Qwen XML protocol. We emit this whether or
    # not `tools` is non-empty, because ReAct does NOT pass tools to the
    # adapter as an input field — it embeds them in signature.instructions
    # text. So the adapter's `tools` input is typically empty for ReAct turns,
    # but the model still needs to see the XML protocol.
    if react_fields:
        sections.append(_REACT_GUIDANCE)
    if task_description:
        sections.append(task_description.strip())
    if tool_list:
        block = "<tools>\n" + "\n".join(_tool_to_xml(t) for t in tool_list) + "\n</tools>"
        sections.append(block)
    if tool_list and not react_fields:
        sections.append(_EXEMPLAR)
        sections.append(
            "Emit exactly one <function=...>...</function> per turn, or answer in plain text when done."
        )
    return "\n\n".join(sections)
