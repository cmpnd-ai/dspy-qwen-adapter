import re

_BALANCED_THINK = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_ORPHAN_CLOSER = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_OPENER = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    text = _BALANCED_THINK.sub("", text)
    text = _ORPHAN_CLOSER.sub("", text, count=1)
    text = _UNCLOSED_OPENER.sub("", text)
    return text.strip()
