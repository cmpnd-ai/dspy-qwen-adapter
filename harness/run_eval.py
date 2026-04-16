"""ReAct harness runner.

Invokes dspy.ReAct against a chosen adapter + scenario, collects per-run
metrics, and writes a CSV to harness/results/. Optionally dumps raw LM
content to harness/traces/ for fixture promotion.

Example:
    python -m harness.run_eval --adapter qwen35 --scenario s1 --runs 20
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import dspy
from dspy.utils.callback import BaseCallback

from dspy_qwen35_adapter import Qwen35Adapter
from harness.scenarios import ALL_SCENARIOS, Scenario


RESULTS_DIR = Path(__file__).parent / "results"
TRACES_DIR = Path(__file__).parent / "traces"

CSV_FIELDS = [
    "scenario",
    "adapter",
    "run_idx",
    "turns_completed",
    "max_iters_hit",
    "parse_failures",
    "tool_exec_failures",
    "task_succeeded",
    "error",
]


# -------- Adapter construction --------

def build_adapter(name: str):
    """Instantiate a fresh adapter by short name."""
    if name == "chat":
        return dspy.ChatAdapter()
    if name == "json":
        return dspy.JSONAdapter()
    if name == "qwen35":
        return Qwen35Adapter()
    raise ValueError(f"unknown adapter: {name}")


# -------- Callbacks --------

class ParseFailureCounter(BaseCallback):
    """Counts adapter.parse() calls that raised an exception."""

    def __init__(self) -> None:
        self.parse_failures = 0

    def on_adapter_parse_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        if exception is not None:
            self.parse_failures += 1


class TraceCapture(BaseCallback):
    """Writes raw LM completion content to disk, one file per LM call.

    File name: {scenario}-{adapter}-r{run}-t{turn}.txt
    """

    def __init__(self, scenario: str, adapter: str, run_idx: int, out_dir: Path) -> None:
        self.scenario = scenario
        self.adapter = adapter
        self.run_idx = run_idx
        self.out_dir = out_dir
        self.turn = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_lm_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ) -> None:
        if outputs is None:
            return
        # `dspy.LM` returns a list of strings (content) or list of dicts
        # (with a "text" key) depending on response format. Normalize both.
        try:
            if isinstance(outputs, list):
                pieces = []
                for item in outputs:
                    if isinstance(item, str):
                        pieces.append(item)
                    elif isinstance(item, dict):
                        pieces.append(item.get("text") or item.get("content") or str(item))
                    else:
                        pieces.append(str(item))
                content = "\n---\n".join(pieces)
            else:
                content = str(outputs)
        except Exception:
            content = repr(outputs)

        fname = (
            f"{self.scenario}-{self.adapter}-r{self.run_idx}-t{self.turn}.txt"
        )
        path = self.out_dir / fname
        path.write_text(content, encoding="utf-8")
        self.turn += 1


# -------- Per-run metric collection --------

def _count_turns(trajectory: dict[str, Any]) -> int:
    return sum(1 for k in trajectory if k.startswith("tool_name_"))


def _count_tool_exec_failures(trajectory: dict[str, Any]) -> int:
    n = 0
    for k, v in trajectory.items():
        if k.startswith("observation_") and isinstance(v, str) and "Execution error" in v:
            n += 1
    return n


def _check_success(answer: Any, golden: str) -> bool:
    if not isinstance(answer, str):
        answer = str(answer) if answer is not None else ""
    return golden.lower() in answer.lower()


def _build_signature(question_field_name: str = "question"):
    """Simple ReAct-friendly signature: question -> answer."""
    return dspy.Signature("question -> answer", "Answer the question using the tools.")


def run_once(
    scenario: Scenario,
    adapter_name: str,
    run_idx: int,
    max_iters: int,
    capture_traces: bool,
) -> dict[str, Any]:
    """Execute one ReAct run and return a metrics dict."""
    counter = ParseFailureCounter()
    callbacks: list[BaseCallback] = [counter]
    if capture_traces:
        callbacks.append(
            TraceCapture(
                scenario=scenario.name,
                adapter=adapter_name,
                run_idx=run_idx,
                out_dir=TRACES_DIR,
            )
        )

    # Fresh adapter per run so any internal state doesn't leak.
    adapter = build_adapter(adapter_name)

    row: dict[str, Any] = {
        "scenario": scenario.name,
        "adapter": adapter_name,
        "run_idx": run_idx,
        "turns_completed": 0,
        "max_iters_hit": False,
        "parse_failures": 0,
        "tool_exec_failures": 0,
        "task_succeeded": False,
        "error": "",
    }

    try:
        with dspy.context(adapter=adapter, callbacks=callbacks):
            signature = _build_signature()
            react = dspy.ReAct(signature, tools=list(scenario.tools), max_iters=max_iters)
            pred = react(question=scenario.question)

        trajectory = getattr(pred, "trajectory", {}) or {}
        turns = _count_turns(trajectory)
        row["turns_completed"] = turns
        # max_iters_hit: we made max_iters turns AND the last turn wasn't "finish".
        last_tool = trajectory.get(f"tool_name_{max_iters - 1}")
        row["max_iters_hit"] = turns >= max_iters and last_tool != "finish"
        row["tool_exec_failures"] = _count_tool_exec_failures(trajectory)
        row["task_succeeded"] = _check_success(
            getattr(pred, "answer", None), scenario.golden_answer_substring
        )
    except Exception as e:  # pragma: no cover - only fires on real LM failures
        row["error"] = type(e).__name__
        # Print traceback to stderr so live debugging stays possible.
        traceback.print_exc(file=sys.stderr)

    # parse_failures is accumulated via callback regardless of success/error.
    row["parse_failures"] = counter.parse_failures
    return row


# -------- CLI + main --------

def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="harness.run_eval",
        description="Run a DSPy ReAct scenario against a chosen adapter and log metrics.",
    )
    p.add_argument(
        "--adapter",
        required=True,
        choices=["chat", "json", "qwen35"],
        help="Which adapter to evaluate.",
    )
    p.add_argument(
        "--scenario",
        required=True,
        choices=sorted(ALL_SCENARIOS.keys()),
        help="Which scenario to run.",
    )
    p.add_argument("--runs", type=int, default=20, help="Number of runs (default 20).")
    p.add_argument(
        "--max-iters",
        type=int,
        default=10,
        help="ReAct max_iters per run (default 10).",
    )
    p.add_argument(
        "--capture-traces",
        action="store_true",
        help="Dump raw LM content to harness/traces/ for each turn.",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", "openai/qwen/qwen3.5-35b-a3b"),
        help="LM model id (default: $QWEN_MODEL or openai/qwen/qwen3.5-35b-a3b).",
    )
    p.add_argument(
        "--api-base",
        default=os.environ.get("LMSTUDIO_BASE", "http://127.0.0.1:1234/v1"),
        help="OpenAI-compatible API base URL.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LM sampling temperature (default 0.0).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="LM max_tokens per call (default 8192).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Override the CSV output path.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    scenario = ALL_SCENARIOS[args.scenario]

    # Configure the LM once. dspy.context(adapter=...) rebinds per-run.
    lm = dspy.LM(
        model=args.model,
        api_base=args.api_base,
        api_key="lm-studio",  # LiteLLM wants a non-empty value; LM Studio ignores.
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache=False,  # each run must hit the model fresh.
    )
    dspy.configure(lm=lm)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.output)
        if args.output
        else RESULTS_DIR / f"{_timestamp()}-{args.adapter}-{args.scenario}.csv"
    )

    print(
        f"[harness] adapter={args.adapter} scenario={scenario.name} runs={args.runs} "
        f"model={args.model} api_base={args.api_base}"
    )
    print(f"[harness] writing CSV -> {csv_path}")
    if args.capture_traces:
        print(f"[harness] capturing traces -> {TRACES_DIR}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for i in range(args.runs):
            row = run_once(
                scenario=scenario,
                adapter_name=args.adapter,
                run_idx=i,
                max_iters=args.max_iters,
                capture_traces=args.capture_traces,
            )
            writer.writerow(row)
            f.flush()
            print(
                f"run {i}: turns={row['turns_completed']} "
                f"parse_fail={row['parse_failures']} "
                f"tool_fail={row['tool_exec_failures']} "
                f"ok={row['task_succeeded']}"
                + (f" err={row['error']}" if row["error"] else "")
            )

    print(f"[harness] done. {args.runs} rows -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
