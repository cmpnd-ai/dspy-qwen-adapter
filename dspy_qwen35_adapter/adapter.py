import logging
from typing import Any

from dspy.adapters.base import Adapter
from dspy.adapters.types.tool import Tool
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback

from dspy_qwen35_adapter.parsing import (
    strip_think,
    split_thought_and_call,
    coerce_args_to_schema,
)
from dspy_qwen35_adapter.prompts import build_system_prompt

logger = logging.getLogger(__name__)

REACT_TOOL_FIELDS = {"next_thought", "next_tool_name", "next_tool_args"}

# Phrases in ReAct's auto-generated signature.instructions that push the model
# toward the wrong output format for this adapter. We strip lines containing
# any of these before prepending the instructions to our Qwen-native system
# prompt. Other ReAct framing (agent role, goal, tool enumeration) is kept.
_REACT_OUTPUT_FORMAT_CUES = (
    "JSON", "json",
    "next_thought", "next_tool_name", "next_tool_args",
)


def _scrub_react_format_directives(instructions: str) -> str:
    """Remove lines that reference ReAct's field-named output protocol.

    ReAct's generated instructions tell the model to "interleave next_thought,
    next_tool_name, and next_tool_args" and to format args "in JSON format".
    Qwen anchors on those cues and produces JSON or pseudo-YAML. Dropping
    those lines lets our _REACT_GUIDANCE (appended later in the system
    prompt by build_system_prompt) be the sole authority on output format.
    """
    if not instructions:
        return ""
    return "\n".join(
        ln for ln in instructions.splitlines()
        if not any(cue in ln for cue in _REACT_OUTPUT_FORMAT_CUES)
    )


class Qwen35Adapter(Adapter):
    """Text-native DSPy adapter for Qwen 3.5 tool calling.

    Bypasses the inference server's tool-call parser entirely. Prompts Qwen
    in its trained XML format and parses tool calls out of the assistant's
    content directly.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        native_response_types: list[type] | None = None,
        strict_parse: bool = False,
    ):
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=False,
            native_response_types=native_response_types,
        )
        self.strict_parse = strict_parse

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        cleaned = strip_think(completion)
        thought, call = split_thought_and_call(cleaned)

        has_react_fields = REACT_TOOL_FIELDS.issubset(signature.output_fields.keys())

        if has_react_fields:
            if call is None:
                if self.strict_parse:
                    from dspy.utils.exceptions import AdapterParseError
                    raise AdapterParseError(
                        adapter_name="Qwen35Adapter",
                        signature=signature,
                        lm_response=completion,
                        message="No <function=...> tool call found.",
                    )
                return {
                    "next_thought": thought,
                    "next_tool_name": "finish",
                    "next_tool_args": {},
                }
            name, args = call
            return {
                "next_thought": thought,
                "next_tool_name": name,
                "next_tool_args": args,
            }

        # Non-ReAct signatures: best-effort plain-text → last output field.
        # ChainOfThought prepends 'reasoning'; user's real answer is last.
        output_keys = list(signature.output_fields.keys())
        if not output_keys:
            return {}
        result = {k: "" for k in output_keys}
        result[output_keys[-1]] = cleaned
        return result

    def format_field_description(self, signature: type[Signature]) -> str:
        return signature.instructions or ""

    def format_field_structure(self, signature: type[Signature]) -> str:
        return (
            "When calling a tool, emit exactly one <function=NAME>"
            "<parameter=KEY>VALUE</parameter>...</function>. "
            "Otherwise respond in plain text."
        )

    def format_task_description(self, signature: type[Signature]) -> str:
        return signature.instructions or ""

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        parts = []
        for name, _field in signature.input_fields.items():
            if name == "tools":
                continue
            value = inputs.get(name, "")
            parts.append(f"{name}: {value}")
        return (prefix + "\n\n".join(parts) + suffix).strip()

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        has_react_fields = REACT_TOOL_FIELDS.issubset(signature.output_fields.keys())
        if has_react_fields:
            thought = outputs.get("next_thought", "")
            name = outputs.get("next_tool_name", "")
            args = outputs.get("next_tool_args", {}) or {}
            params = "".join(
                f"<parameter={k}>{v}</parameter>" for k, v in args.items()
            )
            return f"{thought}\n<function={name}>{params}</function>".strip()
        return "\n".join(f"{k}: {v}" for k, v in outputs.items())

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        tools = inputs.get("tools") or []
        tool_list = tools if isinstance(tools, list) else [tools]
        react_fields = REACT_TOOL_FIELDS.issubset(signature.output_fields.keys())
        task_description = _scrub_react_format_directives(
            signature.instructions or ""
        ) if react_fields else (signature.instructions or "")
        system = build_system_prompt(
            task_description=task_description,
            tools=tool_list,
            react_fields=react_fields,
        )
        user = self.format_user_message_content(signature, inputs)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
