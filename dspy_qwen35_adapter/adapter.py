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


_TRAJECTORY_PREFIXES = ("thought_", "tool_name_", "tool_args_", "observation_")


def _render_tool_call(name: str, args: dict[str, Any]) -> str:
    """Render a (tool_name, args) pair in Qwen 3.5's canonical chat-template
    XML format: <tool_call><function=NAME><parameter=K>\\nVALUE\\n</parameter>
    ...</function></tool_call> with values on their own lines.

    Mirrors the tokenizer_config.json chat_template for Qwen 3.5 (and the
    Qwen3-Coder lineage it inherits from)."""
    params = "".join(
        f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items()
    )
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def _is_react_trajectory(inputs: dict[str, Any]) -> bool:
    """True when the inputs dict looks like a ReAct trajectory (keys of the
    form thought_N / tool_name_N / tool_args_N / observation_N). ReAct's
    _format_trajectory passes the trajectory dict here to get a text blob
    for the next turn's user message."""
    if not inputs:
        return False
    return any(str(k).startswith(_TRAJECTORY_PREFIXES) for k in inputs)


def _render_react_trajectory(trajectory: dict[str, Any]) -> str:
    """Render a ReAct trajectory as a Qwen-native transcript: each turn's
    reasoning as plain text, the tool call in canonical XML, and the
    observation wrapped in <tool_response>.

    Turns are grouped by the integer suffix on the keys (thought_0,
    tool_name_0, etc.). Missing pieces of a turn are silently skipped.

    The <tool_response> tag carries a `name` attribute identifying which
    tool produced the response. This is a slight deviation from Qwen 3.5's
    bare canonical chat template (which does not include attributes on
    tool_response). We add it because small models (e.g. Qwen 3.5 4B)
    otherwise lose track of which tool produced which output on the
    extract turn, and paraphrase tool outputs away when constructing the
    final answer. Tested empirically — see the `s_i18n` 4B regression
    notes in docs/benchmarks.md."""
    turn_idxs: list[int] = sorted({
        int(k.rsplit("_", 1)[1])
        for k in trajectory
        if str(k).startswith(_TRAJECTORY_PREFIXES)
        and k.rsplit("_", 1)[1].isdigit()
    })
    parts: list[str] = []
    for i in turn_idxs:
        thought = trajectory.get(f"thought_{i}", "")
        name = trajectory.get(f"tool_name_{i}", "")
        args = trajectory.get(f"tool_args_{i}", {}) or {}
        obs = trajectory.get(f"observation_{i}", None)
        if thought:
            parts.append(str(thought).strip())
        if name:
            parts.append(_render_tool_call(name, args))
        if obs is not None:
            attr = f' name="{name}"' if name else ""
            parts.append(f"<tool_response{attr}>\n{obs}\n</tool_response>")
    return "\n".join(parts)


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

    def _call_postprocess(
        self,
        processed_signature,
        original_signature,
        outputs,
        lm,
        lm_kwargs,
    ):
        """Rescue turns where LM Studio / vLLM reasoning-parsers routed
        everything into `reasoning_content` and left `text` empty.

        Qwen 3 and 3.5 models in "thinking mode" may emit so much <think>
        content that the server's reasoning parser consumes the entire
        completion — leaving `text=""`. DSPy's base _call_postprocess then
        skips `parse()` entirely (base.py:136) and returns all output fields
        as None, which kills the ReAct / extract turn.

        We merge reasoning_content into text as a fallback so our parser
        still has something to work with — often the runaway reasoning
        contains a usable tool-call or final answer at the end.
        """
        normalized = []
        for output in outputs:
            if (
                isinstance(output, dict)
                and not output.get("text")
                and output.get("reasoning_content")
            ):
                output = {**output, "text": output["reasoning_content"]}
                logger.debug(
                    "Qwen35Adapter: text was empty; promoted reasoning_content "
                    "(%d chars) into text for parsing.",
                    len(output["text"]),
                )
            normalized.append(output)
        return super()._call_postprocess(
            processed_signature, original_signature, normalized, lm, lm_kwargs
        )

    def format_field_description(self, signature: type[Signature]) -> str:
        return signature.instructions or ""

    def format_field_structure(self, signature: type[Signature]) -> str:
        return (
            "When calling a tool, emit a canonical Qwen tool call: "
            "<tool_call><function=NAME><parameter=KEY>VALUE</parameter>..."
            "</function></tool_call>. Otherwise respond in plain text."
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
        # ReAct calls format_user_message_content(trajectory_signature, trajectory)
        # to render the past trajectory as text to feed back to the model.
        # When we detect that pattern, render in Qwen's canonical chat-template
        # format: <tool_call>...</tool_call> + <tool_response>...</tool_response>.
        if _is_react_trajectory(inputs):
            rendered = _render_react_trajectory(inputs)
            return (prefix + rendered + suffix).strip()

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
            call_xml = _render_tool_call(name, args)
            if thought:
                return f"{thought}\n{call_xml}"
            return call_xml
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
        # Extract-turn guidance: when the signature has a pre-rendered
        # `trajectory` input but is NOT a ReAct turn (so it's the
        # ChainOfThought extract that produces the final answer), small
        # models tend to paraphrase tool outputs away. Asking explicitly
        # for verbatim reporting keeps them from cleaning up prefixes or
        # structural markers that may carry task-relevant information.
        is_extract_turn = (
            not react_fields
            and "trajectory" in signature.input_fields
        )
        if is_extract_turn:
            system = (
                (system + "\n\n" if system else "")
                + "When your answer references information obtained from a "
                "tool call in the trajectory, quote the tool's output "
                "verbatim rather than paraphrasing or summarizing it."
            )
        user = self.format_user_message_content(signature, inputs)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
