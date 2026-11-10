import dataclasses
import json
import logging
from contextvars import ContextVar
from typing import Any, get_origin

from dspy.adapters.base import Adapter as _BaseAdapter
from dspy.adapters.base import Prompt
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.lm15 import Message, TextPart, ThinkingPart
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError

from dspy_qwen_adapter.parsing import (
    decode_for_schema,
    split_thought_and_calls,
    strip_think,
)
from dspy_qwen_adapter.prompts import build_system_prompt

logger = logging.getLogger(__name__)

# DSPy's `Adapter.__init_subclass__` wraps `format` and `parse` with
# `with_callbacks` on every subclass, so inherited methods arrive here already
# wrapped once per ancestor. Calling the unwrapped implementations keeps one
# callback span per call. XMLAdapter's hooks still resolve through `self`.
_ADAPTER_FORMAT = _BaseAdapter.format  # unwrapped; Adapter itself is never a subclass
_XML_PARSE_UNWRAPPED = XMLAdapter.parse.__wrapped__

# Tools for the in-flight call, keyed by name. `parse()` reads their schemas to
# decode <parameter> values. Set in `_call_preprocess`, which sees the inputs.
_ACTIVE_TOOLS: ContextVar[dict[str, Tool]] = ContextVar("qwen_adapter_active_tools", default={})

# Tool named by a `tool_choice` option for the in-flight call (e.g. ReAct's
# forced `submit`). The adapter never sends tools to the server, so it
# enforces the choice in the prompt instead.
_FORCED_TOOL: ContextVar[str | None] = ContextVar("qwen_adapter_forced_tool", default=None)


def _forced_tool_name(tool_choice: Any) -> str | None:
    """The function name a `tool_choice` value requires, if any."""
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            return function["name"]
        return tool_choice.get("name")
    return None


def _tool_calls_field(signature: type[Signature]) -> str | None:
    """Name of the `dspy.ToolCalls` output field, if the signature has one."""
    for name, field in signature.output_fields.items():
        if field.annotation == ToolCalls:
            return name
    return None


def _tools_field(signature: type[Signature]) -> str | None:
    """Name of the `list[dspy.Tool]` or `dspy.Tool` input field, if any."""
    for name, field in signature.input_fields.items():
        if field.annotation == Tool:
            return name
        if get_origin(field.annotation) is list and field.annotation.__args__[0] == Tool:
            return name
    return None


def _is_tool_signature(signature: type[Signature]) -> bool:
    """True for signatures that ask the model to call tools (e.g. `dspy.ReAct`)."""
    return _tool_calls_field(signature) is not None


def _thought_field(signature: type[Signature]) -> str | None:
    """The output field that carries prose reasoning on a tool-calling turn."""
    tc_field = _tool_calls_field(signature)
    for name in signature.output_fields:
        if name != tc_field:
            return name
    return None


def _as_tool_list(value: Any) -> list[Tool]:
    if not value:
        return []
    return list(value) if isinstance(value, list) else [value]


def _render_tool_call(name: str, args: dict[str, Any]) -> str:
    """Render one call in Qwen 3.5's chat-template XML: values sit on their own
    lines, strings verbatim and everything else as JSON."""

    def fmt(v: Any) -> str:
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

    params = "".join(f"<parameter={k}>\n{fmt(v)}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def _render_tool_responses(results: ToolCallResults) -> str:
    """Render tool results as Qwen <tool_response> blocks. The `name` attribute
    is an addition to Qwen's bare template that keeps provenance clear when a
    turn has several results."""
    blocks = []
    for result in results.tool_call_results:
        value = result.value
        if not isinstance(value, str):
            try:
                value = json.dumps(value, ensure_ascii=False, default=str)
            except TypeError:
                value = str(value)
        blocks.append(f'<tool_response name="{result.name}">\n{value}\n</tool_response>')
    return "\n".join(blocks)


def _merge_user_turns(messages: tuple[Message, ...]) -> tuple[Message, ...]:
    """Join adjacent text-only user messages. ReAct history yields a tool-result
    turn followed by the next turn's inputs; Qwen expects a single user turn."""
    merged: list[Message] = []
    for msg in messages:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.role == "user"
            and msg.role == "user"
            and all(isinstance(p, TextPart) for p in prev.parts + msg.parts)
        ):
            text = "\n\n".join(p.text for p in prev.parts + msg.parts)
            merged[-1] = Message.user(text)
        else:
            merged.append(msg)
    return tuple(merged)


def _thinking_text(response: Any) -> str:
    """The model's reasoning text.

    Over the OpenAI Responses API, LM Studio returns Qwen's reasoning in
    `content[].reasoning_text` with an empty `summary`, and lm15 only maps the
    summary into `ThinkingPart`. Fall back to the raw output items then.
    """
    thinking = "".join(p.text for p in response.message.parts if isinstance(p, ThinkingPart))
    if thinking:
        return thinking
    output = (response.provider_data or {}).get("output")
    if not isinstance(output, list):
        return ""
    return "".join(
        block.get("text", "")
        for item in output
        if isinstance(item, dict) and item.get("type") == "reasoning"
        for block in item.get("content") or []
        if isinstance(block, dict) and block.get("type") == "reasoning_text"
    )


def _holds_answer(thinking: str, signature: type[Signature]) -> bool:
    """True when the thinking text carries a usable answer, not just
    deliberation that happens to mention the output tags."""
    if _is_tool_signature(signature):
        return "<function=" in thinking or "<tool_call>" in thinking
    fields = list(signature.output_fields)
    return bool(fields) and thinking.rstrip().endswith(f"</{fields[-1]}>")


def _rescue_reasoning(response: Any, signature: type[Signature]) -> Any:
    """Move answer text out of the thinking channel when the server's
    reasoning parser put it there.

    Two cases:
    1. No visible text: the parser consumed the whole completion. Promote the
       thinking text so `parse()` has something to read, but only when it
       holds an answer. A model that ran out of tokens mid-deliberation should
       fail to parse, not yield fields scraped from its reasoning.
    2. Tool-calling turn whose visible text has no call while the thinking
       text does: append the thinking text so the call can be found.
    """
    parts = response.message.parts
    text = "".join(p.text for p in parts if isinstance(p, TextPart))
    thinking = _thinking_text(response)
    tool_signature = _is_tool_signature(signature)
    if not thinking or not _holds_answer(thinking, signature):
        return response

    if not text.strip():
        new_text = thinking
        logger.debug("QwenAdapter: text was empty; promoted %d chars of thinking.", len(thinking))
    elif tool_signature and "<function=" not in text and "<function=" in thinking:
        new_text = text + "\n" + thinking
        logger.debug("QwenAdapter: tool call found only in thinking; appended %d chars.", len(thinking))
    else:
        return response

    kept = tuple(p for p in parts if not isinstance(p, TextPart))
    message = Message(role="assistant", parts=kept + (TextPart(new_text),))
    return dataclasses.replace(response, message=message)


class QwenAdapter(XMLAdapter):
    """DSPy adapter that keeps Qwen 3.5 in its trained distribution for both
    tool-calling and plain Predict signatures.

    Tool-calling signatures (any signature with a `dspy.ToolCalls` output,
    including `dspy.ReAct`) get a Qwen-native prompt: tools listed in a
    <tools> block, past calls replayed as `<tool_call><function=...>` XML,
    results wrapped in `<tool_response>`, and calls parsed from the visible
    text. The server's native function-calling path is never used.

    Other signatures use `XMLAdapter`'s `<field>content</field>` protocol.

    Both paths strip leaked `<think>` blocks and recover completions that the
    server routed into the reasoning channel. With `strict_parse=False`
    (default), a tool-calling turn with no call returns an empty `ToolCalls`,
    which makes ReAct force a final `submit`. With `strict_parse=True` it
    raises `AdapterParseError` instead.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        native_response_types: list[type] | None = None,
        strict_parse: bool = False,
    ):
        super().__init__(callbacks=callbacks)
        # The adapter never sends tools to the server, whatever the provider
        # claims to support.
        self.use_native_function_calling = False
        self.use_json_adapter_fallback = False
        if native_response_types is not None:
            self.native_response_types = native_response_types
        self.strict_parse = strict_parse

    # -------- call pipeline --------

    def _call_preprocess(self, lm, lm_kwargs, signature, inputs):
        tools_field = _tools_field(signature)
        tools = _as_tool_list(inputs.get(tools_field)) if tools_field else []
        _ACTIVE_TOOLS.set({t.name: t for t in tools})
        _FORCED_TOOL.set(_forced_tool_name(lm_kwargs.get("tool_choice")))
        return super()._call_preprocess(lm, lm_kwargs, signature, inputs)

    def _call_postprocess(self, processed_signature, original_signature, responses):
        responses = [_rescue_reasoning(r, processed_signature) for r in responses]
        return super()._call_postprocess(processed_signature, original_signature, responses)

    # -------- format --------

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> Prompt:
        prompt = _ADAPTER_FORMAT(self, signature, demos, inputs)
        if not _is_tool_signature(signature):
            return prompt

        tools_field = _tools_field(signature)
        tools = _as_tool_list(inputs.get(tools_field)) if tools_field else []
        forced = _FORCED_TOOL.get()
        if forced and any(t.name == forced for t in tools):
            tools = [t for t in tools if t.name == forced]
        system = build_system_prompt(
            task_description=signature.instructions or "",
            tools=tools,
            react_fields=True,
            required_tool=forced if tools and tools[0].name == forced else None,
        )
        return Prompt(system=system, messages=_merge_user_turns(prompt.messages))

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        # History replays tool results through a one-field signature.
        results = inputs.get("tool_call_results")
        if list(signature.input_fields) == ["tool_call_results"] and results is not None:
            rendered = _render_tool_responses(ToolCallResults.model_validate(results))
            return (prefix + rendered + suffix).strip()

        if _is_tool_signature(signature):
            # Tools live in the system prompt and history is replayed as
            # turns, so neither belongs in the user message.
            for name, field in list(signature.input_fields.items()):
                if name == _tools_field(signature) or name == "history":
                    signature = signature.delete(name)
            inputs = {k: v for k, v in inputs.items() if k in signature.input_fields}
            if not inputs:
                return ""
            # XMLAdapter's "Respond with <field>..." footer contradicts the
            # tool-call protocol in the system prompt.
            main_request = False

        return super().format_user_message_content(
            signature, inputs, prefix=prefix, suffix=suffix, main_request=main_request
        )

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        if _is_tool_signature(signature):
            thought_field = _thought_field(signature)
            thought = outputs.get(thought_field) if thought_field else None
            raw_calls = outputs.get(_tool_calls_field(signature))
            calls = ToolCalls.model_validate(raw_calls).tool_calls if raw_calls else []
            pieces = [str(thought).strip()] if thought else []
            pieces.extend(_render_tool_call(c.name, c.args or {}) for c in calls)
            return "\n".join(pieces)

        return super().format_assistant_message_content(
            signature, outputs, missing_field_message=missing_field_message
        )

    # -------- parse --------

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        cleaned = strip_think(completion)

        if not _is_tool_signature(signature):
            return _XML_PARSE_UNWRAPPED(self, signature, cleaned)

        thought, calls = split_thought_and_calls(cleaned)
        if not calls and self.strict_parse:
            raise AdapterParseError(
                adapter_name="QwenAdapter",
                signature=signature,
                lm_response=completion,
                message="No <function=...> tool call found.",
            )

        tools = _ACTIVE_TOOLS.get()
        decoded = []
        for name, raw_args in calls:
            tool = tools.get(name)
            schemas = (tool.args or {}) if tool else {}
            args = {k: decode_for_schema(v, schemas.get(k)) for k, v in raw_args.items()}
            decoded.append({"name": name, "args": args})

        result: dict[str, Any] = {_tool_calls_field(signature): ToolCalls.from_dict_list(decoded)}
        if thought_field := _thought_field(signature):
            result[thought_field] = thought
        return result
