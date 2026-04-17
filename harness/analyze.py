# harness/analyze.py
"""Summarize all CSVs under harness/results/ as a markdown table comparing
adapters across scenarios. Shows both the cheap substring-match task_success
and the LLM-judge verdict when `--use-judge` was enabled during runs."""
import csv
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def _rate(rows: list[dict], col: str) -> float | None:
    """Fraction of rows where `col == 'True'`. Returns None when every row
    has the column empty (meaning that mode wasn't run)."""
    values = [r.get(col, "") for r in rows]
    non_empty = [v for v in values if v != ""]
    if not non_empty:
        return None
    return sum(v == "True" for v in non_empty) / len(non_empty)


def main():
    cells = defaultdict(list)
    for path in sorted(RESULTS_DIR.glob("*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["scenario"], row["adapter"])
                cells[key].append(row)

    scenarios = sorted({k[0] for k in cells})
    adapters = sorted({k[1] for k in cells})

    # Detect whether any run used the judge — if so, show a judge_pass column.
    any_judged = any(
        r.get("judge_pass", "") != ""
        for rows in cells.values()
        for r in rows
    )

    if any_judged:
        print("| scenario | adapter | runs | parse_fail / run | tool_fail / run | task_success (substring) | judge_pass |")
        print("|---|---|---|---|---|---|---|")
    else:
        print("| scenario | adapter | runs | parse_fail / run | tool_fail / run | task_success |")
        print("|---|---|---|---|---|---|")

    for s in scenarios:
        for a in adapters:
            rows = cells.get((s, a), [])
            if not rows:
                continue
            n = len(rows)
            pf = sum(int(r["parse_failures"]) for r in rows) / n
            tf = sum(int(r["tool_exec_failures"]) for r in rows) / n
            ok = _rate(rows, "task_succeeded") or 0.0

            if any_judged:
                judge = _rate(rows, "judge_pass")
                judge_str = f"{judge:.0%}" if judge is not None else "—"
                print(f"| {s} | {a} | {n} | {pf:.2f} | {tf:.2f} | {ok:.0%} | {judge_str} |")
            else:
                print(f"| {s} | {a} | {n} | {pf:.2f} | {tf:.2f} | {ok:.0%} |")


if __name__ == "__main__":
    main()
