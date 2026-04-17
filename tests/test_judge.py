"""Unit tests for the LLM judge. Only the short-circuit paths are covered
here — full judge behavior requires a live LM and is validated via the
harness smoke runs."""
import pytest

from harness.judge import judge_answer


def test_empty_criterion_short_circuits_to_false():
    """A scenario without a judge_criterion must not invoke the LM at all."""
    verdict, reason = judge_answer(
        question="anything",
        criterion="",
        answer="anything",
        judge_lm=None,  # must not be dereferenced
    )
    assert verdict is False
    assert "no judge criterion" in reason


def test_empty_answer_short_circuits_to_false():
    """An empty pred.answer fails without burning a judge call."""
    verdict, reason = judge_answer(
        question="anything",
        criterion="the answer must be nonempty",
        answer="",
        judge_lm=None,
    )
    assert verdict is False
    assert "empty" in reason


def test_whitespace_answer_short_circuits_to_false():
    """Whitespace-only answers are treated as empty."""
    verdict, reason = judge_answer(
        question="anything",
        criterion="criterion here",
        answer="   \n\t   ",
        judge_lm=None,
    )
    assert verdict is False


def test_none_answer_short_circuits_to_false():
    """None answers (possible if extract fails entirely) don't crash."""
    verdict, reason = judge_answer(
        question="anything",
        criterion="criterion here",
        answer="",  # run_eval converts None -> "" before calling judge
        judge_lm=None,
    )
    assert verdict is False
