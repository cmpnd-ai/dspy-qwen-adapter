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

logger = logging.getLogger(__name__)

REACT_TOOL_FIELDS = {"next_thought", "next_tool_name", "next_tool_args"}


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

        # Non-ReAct signatures fall through to the XML/delimiter fallback
        # added in Task 10. For now, return thought as best-effort.
        return {list(signature.output_fields.keys())[0]: cleaned}
