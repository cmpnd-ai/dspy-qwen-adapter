import re

_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>")
_ORPHAN_CLOSER = re.compile(r"^\s*</think>\s*")
_UNCLOSED_THINK = re.compile(r"<think>[\s\S]*$")


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks, orphan </think> openers, and
    unclosed <think> tails. Whitespace around removed regions is trimmed."""
    text = _ORPHAN_CLOSER.sub("", text)
    text = _THINK_BLOCK.sub("", text)
    text = _UNCLOSED_THINK.sub("", text)
    return text.strip()
