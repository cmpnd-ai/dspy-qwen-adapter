# harness/analyze.py
"""Summarize all CSVs under harness/results/ as a markdown table comparing
adapters across scenarios on parse-failure rate and task success rate."""
import csv
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    cells = defaultdict(list)
    for path in sorted(RESULTS_DIR.glob("*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["scenario"], row["adapter"])
                cells[key].append(row)

    scenarios = sorted({k[0] for k in cells})
    adapters = sorted({k[1] for k in cells})

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
            ok = sum(r["task_succeeded"] == "True" for r in rows) / n
            print(f"| {s} | {a} | {n} | {pf:.2f} | {tf:.2f} | {ok:.0%} |")


if __name__ == "__main__":
    main()
