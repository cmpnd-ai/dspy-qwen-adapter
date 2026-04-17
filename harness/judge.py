"""LLM judge for benchmark answer quality.

Replaces the `golden_answer_substring.lower() in pred.answer.lower()`
heuristic with a boolean pass/fail decision from a judge LM, scored against
a plain-English criterion supplied by each scenario.

The judge deliberately uses `dspy.ChatAdapter` regardless of which adapter
the answer came from — we want a stable, simple prompt shape for the judge
itself, independent of the adapter-under-test.

Caveat: if the judge LM is the same model as the one being benchmarked,
the scores are self-consistent but not independently validated. Prefer a
stronger or different judge model when possible (set `$JUDGE_MODEL` or
pass `--judge-model` to `run_eval.py`).
"""
from __future__ import annotations

import dspy


class JudgeSignature(dspy.Signature):
    """Judge whether a candidate answer correctly addresses a question.

    Read the criterion carefully — it describes what a correct answer MUST
    convey, not the exact wording. Paraphrases that communicate the same
    facts should pass. Answers that omit required information, hedge
    incorrectly, or fail to invoke the expected tool output should fail.

    Return `pass_` as True ONLY when the answer meets every requirement in
    the criterion. Return a one-sentence reason explaining the verdict.
    """

    question: str = dspy.InputField(desc="The question posed to the agent")
    criterion: str = dspy.InputField(
        desc="What a correct answer MUST convey; the rubric for grading"
    )
    answer: str = dspy.InputField(desc="The agent's final answer to grade")
    pass_: bool = dspy.OutputField(desc="True iff the answer meets the criterion")
    reason: str = dspy.OutputField(desc="One sentence explaining the verdict")


def judge_answer(
    question: str,
    criterion: str,
    answer: str,
    judge_lm: dspy.LM,
) -> tuple[bool, str]:
    """Score a single (question, answer) pair against the given criterion.

    Uses ChatAdapter for the judge call regardless of the adapter under
    test — the judge's prompt/parse path should be a constant, not a
    variable, across benchmark rows.
    """
    if not criterion:
        return False, "no judge criterion specified"

    # Short-circuit on obviously empty answers so we don't burn a judge call.
    if not answer or not str(answer).strip():
        return False, "answer is empty"

    with dspy.context(lm=judge_lm, adapter=dspy.ChatAdapter()):
        pred = dspy.Predict(JudgeSignature)(
            question=question,
            criterion=criterion,
            answer=str(answer),
        )
    verdict = bool(getattr(pred, "pass_", False))
    reason = (getattr(pred, "reason", "") or "").strip().replace("\n", " ")
    return verdict, reason[:300]
