import json
from xml.sax.saxutils import escape, quoteattr
from typing import Iterable
from dspy.adapters.types.tool import Tool


_EXEMPLAR = (
    "Example of a tool call:\n"
    "<tool_call>\n"
    "<function=example_tool>\n"
    "<parameter=arg1>\n"
    "value1\n"
    "</parameter>\n"
    "<parameter=arg2>\n"
    "value2\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


_REACT_GUIDANCE = (
    "You are an agent. Use the supplied tools to answer the user's question. "
    "On each turn, write plain-text reasoning on one or more lines, then emit "
    "exactly one tool call in this canonical Qwen format:\n"
    "<tool_call>\n"
    "<function=TOOL_NAME>\n"
    "<parameter=KEY>\n"
    "VALUE\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
    "Emit nothing after the closing </tool_call>. When you have enough "
    "information to answer, call <tool_call><function=finish></function></tool_call>.\n"
    "Example of a correct turn:\n"
    "I need to check Tokyo's weather.\n"
    "<tool_call>\n"
    "<function=get_weather>\n"
    "<parameter=city>\n"
    "Tokyo\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
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
