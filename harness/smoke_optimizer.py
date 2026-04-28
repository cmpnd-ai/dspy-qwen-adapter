"""Integration smoke test: BootstrapFewShot over a ReAct program + QwenAdapter.

Verifies that few-shot demos injected by the optimizer survive through
QwenAdapter.format() and appear in the formatted prompt during inference.

Two deterministic in-process tools (no network):
  - get_population(city)  — returns canned population figures
  - calculate(expression) — evaluates simple arithmetic via Python's eval()

Trainset: 4 questions with known answers.
Metric:   case-insensitive substring match on the expected answer fragment.

Outcome report:
  DEMOS    how many training traces were bootstrapped as few-shot demos
  FORMAT   whether demos actually expand the message list past [system, user]
  INFER    whether the optimized program answers the held-out question correctly

Usage:
    python -m harness.smoke_optimizer
    python -m harness.smoke_optimizer --model openai/qwen/qwen3.5-4b
    QWEN_MODEL=openai/qwen3.5-4b python -m harness.smoke_optimizer
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import dspy

from dspy_qwen_adapter import QwenAdapter


# ──────────────────────────── tools ────────────────────────────

_POPULATIONS: dict[str, int] = {
    "tokyo": 13_960_000,
    "paris": 2_161_000,
    "new york": 8_336_000,
    "london": 8_982_000,
    "berlin": 3_645_000,
    "sydney": 5_312_000,
}


def get_population(city: str) -> str:
    """Return the approximate population of a city."""
    key = city.strip().lower()
    if key in _POPULATIONS:
        return f"The population of {city.title()} is approximately {_POPULATIONS[key]:,}."
    return f"Population data for '{city}' is not available."


def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression and return the numeric result."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: only arithmetic expressions are supported."
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ──────────────────────────── trainset & metric ────────────────────────────

TRAINSET = [
    dspy.Example(
        question="What is the population of Tokyo?",
        answer="13,960,000",
    ).with_inputs("question"),
    dspy.Example(
        question="What is 347 multiplied by 12?",
        answer="4164",
    ).with_inputs("question"),
    dspy.Example(
        question="How many people live in London?",
        answer="8,982,000",
    ).with_inputs("question"),
    dspy.Example(
        question="What is 512 divided by 16?",
        answer="32",
    ).with_inputs("question"),
]

# Held-out test question (requires two tool calls: two get_population + calculate).
TEST_QUESTION = "What is the combined population of Paris and Berlin?"
TEST_ANSWER_FRAGMENT = "5806000"  # 2,161,000 + 3,645,000


def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    answer = str(getattr(pred, "answer", "") or "").strip()
    return example.answer.replace(",", "") in answer.replace(",", "")


# ──────────────────────────── helpers ────────────────────────────

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


# ──────────────────────────── main ────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="harness.smoke_optimizer")
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", "openai/qwen/qwen3.5-35b-a3b"),
    )
    p.add_argument(
        "--api-base",
        default=os.environ.get("LMSTUDIO_BASE", "http://127.0.0.1:1234/v1"),
    )
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument(
        "--max-bootstrapped-demos",
        type=int,
        default=2,
        help="Max few-shot demos to bootstrap (default 2).",
    )
    args = p.parse_args(argv)

    print(f"[optimizer-smoke] model={args.model}  api_base={args.api_base}")
    print()

    lm = _build_lm(args)
    adapter = QwenAdapter()
    dspy.configure(lm=lm, adapter=adapter)

    # ── 1. Build the base program ──
    react = dspy.ReAct(
        "question -> answer",
        tools=[get_population, calculate],
        max_iters=6,
    )

    # ── 2. Optimize with BootstrapFewShot ──
    print("[1/3] Running BootstrapFewShot on trainset …")
    t0 = time.time()
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=0,  # bootstrap only, no labeled demos
    )
    optimized = optimizer.compile(react, trainset=TRAINSET)
    elapsed = time.time() - t0
    print(f"       done in {elapsed:.1f}s")

    # ── 3. Inspect injected demos ──
    # BootstrapFewShot stores bootstrapped traces as demos on each Predict.
    demos = getattr(optimized.react, "demos", []) or []
    n_demos = len(demos)
    print()
    print("[2/3] Demo check")
    print(f"      bootstrapped demos: {n_demos}")
    if n_demos == 0:
        print("      WARNING: no demos bootstrapped — "
              "the model may have failed every training example.")
    else:
        for i, d in enumerate(demos):
            tool = d.get("next_tool_name", "?")
            thought = _truncate(d.get("next_thought", ""), 60)
            print(f"      demo[{i}]: tool={tool!r}  thought={thought!r}")

    # ── 4. FORMAT check: call format() directly with the bootstrapped demos ──
    # This confirms demos survive the adapter without running a full LM call.
    react_sig = optimized.react.signature
    sample_inputs = {
        "question": TEST_QUESTION,
        "trajectory": "",
    }
    messages_with_demos = adapter.format(react_sig, demos, sample_inputs)
    messages_without = adapter.format(react_sig, [], sample_inputs)
    format_ok = len(messages_with_demos) > len(messages_without)

    print()
    print(f"      format() messages without demos: {len(messages_without)}")
    print(f"      format() messages with demos:    {len(messages_with_demos)}")
    print(f"      demos reach format(): {'YES' if format_ok else 'NO — DROPPED (bug!)'}")

    # ── 5. Run inference on the held-out question ──
    print()
    print("[3/3] Inference on held-out question")
    print(f"      question: {TEST_QUESTION!r}")

    t1 = time.time()
    pred = optimized(question=TEST_QUESTION)
    elapsed2 = time.time() - t1

    answer = str(getattr(pred, "answer", "") or "").strip()
    print(f"      answer:   {_truncate(answer, 80)!r}  ({elapsed2:.1f}s)")

    answer_ok = TEST_ANSWER_FRAGMENT in answer.replace(",", "")

    # ── 6. Report ──
    print()
    print("─" * 50)
    print(f"  DEMOS   bootstrapped: {n_demos}   {'OK' if n_demos > 0 else 'WARN'}")
    print(f"  FORMAT  demos reach format(): {'OK' if format_ok else 'FAIL — DROPPED (bug!)'}")
    print(f"  INFER   answer correct: {'OK' if answer_ok else 'FAIL'}")
    print("─" * 50)

    failed = (not format_ok) or (not answer_ok)
    print()
    if n_demos == 0:
        print("NOTE: 0 demos bootstrapped — DEMOS and FORMAT checks vacuously pass.")
        print("      The model may need a larger trainset or more max_bootstrapped_demos.")
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
