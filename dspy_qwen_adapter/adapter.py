import logging
from typing import Any

from dspy.adapters.base import Adapter as _BaseAdapter
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback

from dspy_qwen_adapter.parsing import (
    strip_think,
    split_thought_and_call,
)
from dspy_qwen_adapter.prompts import build_system_prompt

logger = logging.getLogger(__name__)

# DSPy's `Adapter.__init_subclass__` wraps `format` and `parse` with
# `with_callbacks` on every subclass. That re-wraps inherited methods: by the
# time we reach `QwenAdapter`, `XMLAdapter.format` is already double-wrapped
# (once for ChatAdapter + once for XMLAdapter), and a naive `super().format()`
# call would emit two extra callback spans on top of our own wrapper.
#
# Hold direct references to the unwrapped implementations so we can invoke
# them without dragging the extra span layers along. XMLAdapter's XML-tag
# formatting comes from its overridden hooks (format_field_structure,
# format_field_with_value, etc.), which are still resolved via MRO on `self`
# — so calling `_BaseAdapter.format(self, ...)` produces XML-tagged output
# with only one span per call.
_ADAPTER_FORMAT = _BaseAdapter.format  # unwrapped; Adapter itself is never a subclass
_XML_PARSE_UNWRAPPED = XMLAdapter.parse.__wrapped__

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


def _is_react_signature(signature: type[Signature]) -> bool:
    """True when a signature has ReAct's three augmented output fields."""
    return REACT_TOOL_FIELDS.issubset(signature.output_fields.keys())


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
    tool_response). We add it for debugging / multi-tool provenance; the
    parsing side doesn't depend on it."""
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


class QwenAdapter(XMLAdapter):
    """DSPy adapter for Qwen 3.5 that keeps the model in its trained
    distribution across both tool-calling and plain Predict signatures.

    For ReAct signatures (output fields next_thought / next_tool_name /
    next_tool_args), the adapter takes a custom `format()` path that emits
    Qwen's canonical `<tool_call><function=...></function></tool_call>`
    format, replays trajectories with `<tool_response name=...>` wrapping,
    and parses tool calls out of `message.content` via
    `split_thought_and_call`. The server's native function-call path is
    never invoked (`use_native_function_calling=False`).

    For non-ReAct signatures, the adapter inherits `XMLAdapter`'s
    `<field>content</field>` protocol — a natural fit for Qwen's XML-heavy
    training distribution — with the same `<think>` stripping and
    `reasoning_content` rescue layered on top.

    Defensive behaviors applied to both paths:
      - `strip_think()` scrubs leaked `<think>...</think>` blocks.
      - `_call_postprocess()` promotes `reasoning_content` into `text` on
        turns where the server routed the entire completion into the
        reasoning side channel.
      - `strict_parse=False` (default) gracefully finishes a ReAct turn
        when no tool call is emitted; `strict_parse=True` raises
        `AdapterParseError` instead.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        native_response_types: list[type] | None = None,
        strict_parse: bool = False,
    ):
        super().__init__(callbacks=callbacks)
        # Hardcode: we never pass tools=[] to the server, regardless of
        # whether LiteLLM claims the provider supports native function
        # calling. This is the design contract of the adapter.
        self.use_native_function_calling = False
        if native_response_types is not None:
            self.native_response_types = native_response_types
        self.strict_parse = strict_parse

    # -------- format() — branch on ReAct vs plain signatures --------

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if _is_react_signature(signature):
            # Custom system prompt optimized for ReAct with Qwen 3.5:
            # scrubs ReAct's JSON directive from the auto-generated
            # instructions, emits _REACT_GUIDANCE, optionally wraps any
            # `tools` input as a <tools> block (usually empty in prod
            # because ReAct embeds tool enumeration in instructions text).
            tools = inputs.get("tools") or []
            tool_list = tools if isinstance(tools, list) else [tools]
            task_description = _scrub_react_format_directives(
                signature.instructions or ""
            )
            system = build_system_prompt(
                task_description=task_description,
                tools=tool_list,
                react_fields=True,
            )
            user = self.format_user_message_content(
                signature, inputs, main_request=True
            )
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

        # Non-ReAct: delegate to the base Adapter.format composition.
        # XMLAdapter's XML-tag behavior (field descriptions, structure,
        # field_with_value) comes from its overridden hooks, which are still
        # resolved via MRO on `self`. Calling `_ADAPTER_FORMAT` directly
        # avoids a duplicate callback span that `super().format()` would
        # emit (see note at top of module).
        return _ADAPTER_FORMAT(self, signature, demos, inputs)

    # -------- hooks shared between paths --------

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        # ReAct's _format_trajectory helper calls this with a synthetic
        # signature whose input fields are trajectory keys (thought_0,
        # tool_name_0, ...). Render as a Qwen-native transcript.
        if _is_react_trajectory(inputs):
            rendered = _render_react_trajectory(inputs)
            return (prefix + rendered + suffix).strip()

        # A `tools` input field on a ReAct main-turn request is a list of
        # Tool objects — skip it so XMLAdapter doesn't try to XML-wrap a
        # list of Python objects. The tools themselves are already
        # described in the system message (via build_system_prompt).
        if "tools" in inputs and "tools" in signature.input_fields:
            signature = signature.delete("tools")
            inputs = {k: v for k, v in inputs.items() if k != "tools"}

        return super().format_user_message_content(
            signature, inputs, prefix=prefix, suffix=suffix, main_request=main_request
        )

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        if _is_react_signature(signature):
            thought = outputs.get("next_thought", "")
            name = outputs.get("next_tool_name", "")
            args = outputs.get("next_tool_args", {}) or {}
            call_xml = _render_tool_call(name, args)
            if thought:
                return f"{thought}\n{call_xml}"
            return call_xml

        # XMLAdapter renders outputs as <field>content</field>.
        return super().format_assistant_message_content(
            signature, outputs, missing_field_message=missing_field_message
        )

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        cleaned = strip_think(completion)

        if _is_react_signature(signature):
            thought, call = split_thought_and_call(cleaned)
            if call is None:
                if self.strict_parse:
                    from dspy.utils.exceptions import AdapterParseError
                    raise AdapterParseError(
                        adapter_name="QwenAdapter",
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

        # Non-ReAct: XMLAdapter parses `<field>content</field>` tags.
        # Call the unwrapped implementation to avoid a duplicate callback
        # span (see note at top of module).
        return _XML_PARSE_UNWRAPPED(self, signature, cleaned)

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
        completion — leaving `text=""`. DSPy's base `_call_postprocess`
        then skips `parse()` entirely and returns all output fields as
        `None`, which kills the ReAct / extract turn.

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
                    "QwenAdapter: text was empty; promoted reasoning_content "
                    "(%d chars) into text for parsing.",
                    len(output["text"]),
                )
            normalized.append(output)
        return super()._call_postprocess(
            processed_signature, original_signature, normalized, lm, lm_kwargs
        )
