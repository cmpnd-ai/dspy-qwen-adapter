"""Quick live-LM smoke tests for QwenAdapter across DSPy module types.

Verifies the adapter produces parseable output when combined with:

- dspy.Predict         — simplest: signature → answer
- dspy.ChainOfThought  — reasoning + answer
- dspy.RLM             — iterative code execution via an in-process REPL

Not a benchmark — each test runs once, reports pass/fail + elapsed time,
and only checks that output fields are populated sensibly. Total runtime
~30-90s against a local LM. Exits non-zero on any failure.

Usage:
    python -m harness.smoke_modules                          # default model
    QWEN_MODEL=openai/qwen3.5-4b python -m harness.smoke_modules
    python -m harness.smoke_modules --model openai/qwen/qwen3-4b
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from typing import Any, Callable

import dspy
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput

from dspy_qwen_adapter import QwenAdapter


# -------- Minimal in-process code interpreter for the RLM smoke test --------

class _SubmitSentinel(Exception):
    """Raised by SUBMIT() to unwind the exec() call. Caught inside execute()."""


class LocalPythonInterpreter:
    """In-process Python interpreter that implements DSPy's CodeInterpreter
    Protocol without requiring Deno/Pyodide. Executes code via `exec()` in
    a persistent namespace; captures stdout; detects `SUBMIT()` calls and
    returns a `FinalOutput`. For smoke-test use only — not sandboxed.
    """

    def __init__(self, tools: dict[str, Callable[..., str]] | None = None) -> None:
        self._tools: dict[str, Callable[..., str]] = dict(tools or {})
        self._ns: dict[str, Any] = {}
        self._final: Any = None

    @property
    def tools(self) -> dict[str, Callable[..., str]]:
        return self._tools

    def start(self) -> None:
        self._final = None

        def _submit(value: Any = None) -> None:
            self._final = FinalOutput(output=value)
            raise _SubmitSentinel()

        self._ns = {
            "__builtins__": __builtins__,
            "SUBMIT": _submit,
            **self._tools,
        }

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        if not self._ns:
            self.start()
        if variables:
            self._ns.update(variables)

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<rlm-smoke>", "exec"), self._ns, self._ns)
        except _SubmitSentinel:
            return self._final
        except Exception as e:
            raise CodeInterpreterError(f"{type(e).__name__}: {e}") from e

        out = buf.getvalue().strip()
        return out or None

    def shutdown(self) -> None:
        self._ns = {}


# -------- Shared helpers --------

def _build_lm(args: argparse.Namespace) -> dspy.LM:
    return dspy.LM(
        model=args.model,
        api_base=args.api_base,
        api_key="lm-studio",
        temperature=0.0,
        max_tokens=args.max_tokens,
        cache=False,
    )


def _truncate(s: str, n: int = 80) -> str:
    s = str(s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# -------- Smoke tests --------

def test_predict(args: argparse.Namespace) -> tuple[bool, str]:
    """dspy.Predict("question -> answer") must produce a non-empty answer."""
    lm = _build_lm(args)
    with dspy.context(lm=lm, adapter=QwenAdapter()):
        pred = dspy.Predict("question -> answer")(
            question="What is the capital of France?"
        )
    answer = (pred.answer or "").strip()
    if answer:
        return True, f"answer={_truncate(answer)!r}"
    return False, f"empty answer (pred={pred})"


def test_chain_of_thought(args: argparse.Namespace) -> tuple[bool, str]:
    """dspy.ChainOfThought must populate BOTH reasoning and answer."""
    lm = _build_lm(args)
    with dspy.context(lm=lm, adapter=QwenAdapter()):
        pred = dspy.ChainOfThought("question -> answer")(
            question="If a shop sells 3 apples for $2, how much do 9 apples cost?"
        )
    reasoning = (pred.reasoning or "").strip()
    answer = (pred.answer or "").strip()
    if reasoning and answer:
        return True, f"answer={_truncate(answer, 40)!r} reasoning_len={len(reasoning)}"
    return False, (
        f"empty fields: reasoning_len={len(reasoning)} answer={_truncate(answer, 40)!r}"
    )


def test_rlm(args: argparse.Namespace) -> tuple[bool, str]:
    """dspy.RLM must produce output after at least one successful iteration.

    Uses the local in-process interpreter (no Deno/WASM dependency). Task
    is intentionally simple — sum a small list — so the LLM should finish
    in one or two iterations even at small model sizes.
    """
    lm = _build_lm(args)
    interpreter = LocalPythonInterpreter()
    with dspy.context(lm=lm, adapter=QwenAdapter()):
        rlm = dspy.RLM(
            "numbers -> total",
            max_iterations=3,
            interpreter=interpreter,
        )
        try:
            pred = rlm(numbers="[1, 2, 3, 4, 5]")
        except Exception as e:
            return False, f"{type(e).__name__}: {_truncate(str(e), 100)}"
    total = (str(getattr(pred, "total", "") or "")).strip()
    if total:
        return True, f"total={_truncate(total, 40)!r}"
    return False, f"empty total (pred={pred})"


# -------- Entry point --------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="harness.smoke_modules")
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", "openai/qwen/qwen3.5-35b-a3b"),
        help="LM model id (default: $QWEN_MODEL or openai/qwen/qwen3.5-35b-a3b).",
    )
    p.add_argument(
        "--api-base",
        default=os.environ.get("LMSTUDIO_BASE", "http://127.0.0.1:1234/v1"),
    )
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--only",
        choices=["predict", "cot", "rlm"],
        help="Run only one module's test.",
    )
    args = p.parse_args(argv)

    print(f"[smoke] model={args.model} api_base={args.api_base}")

    all_tests: list[tuple[str, Callable[[argparse.Namespace], tuple[bool, str]]]] = [
        ("predict", test_predict),
        ("cot", test_chain_of_thought),
        ("rlm", test_rlm),
    ]
    tests = (
        [(k, v) for k, v in all_tests if k == args.only] if args.only else all_tests
    )

    failed: list[str] = []
    for key, fn in tests:
        t0 = time.time()
        try:
            ok, detail = fn(args)
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {_truncate(str(e), 100)}"
        elapsed = time.time() - t0
        status = "PASS" if ok else "FAIL"
        print(f"[smoke] {status:4s} {key:8s} ({elapsed:5.1f}s) — {detail}")
        if not ok:
            failed.append(key)

    print(f"[smoke] {len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
