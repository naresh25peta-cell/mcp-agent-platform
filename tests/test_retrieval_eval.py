"""
Phase 4 tests: the retrieval eval harness and its CI gate.

Everything here runs offline with the deterministic hashing embedder, so it
needs no network, no model download and no API key.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp_agent_platform.evals import retrieval_eval as ev  # noqa: E402
from mcp_agent_platform.servers import document_qa_server as dq  # noqa: E402

THRESHOLDS = json.loads((ev.EVALS_DIR / "thresholds.json").read_text(encoding="utf-8"))


# --- metric unit tests --------------------------------------------------------

def test_first_hit_rank_is_one_based_and_case_insensitive():
    texts = ["nothing here", "Ten Working Days notice", "also ten working days"]
    assert ev.first_hit_rank(texts, "ten working days") == 2


def test_first_hit_rank_none_when_missing():
    assert ev.first_hit_rank(["a", "b"], "zzz") is None


def test_score_hit_and_mrr():
    # ranks: hit at 1, hit at 2, miss, hit at 4  -> hit 3/4, MRR (1 + .5 + 0 + .25)/4
    s = ev.score([1, 2, None, 4])
    assert s["n"] == 4
    assert s["hit"] == pytest.approx(0.75)
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0.25) / 4)


def test_score_empty_is_zero():
    assert ev.score([]) == {"n": 0, "hit": 0.0, "mrr": 0.0}


# --- gold-set integrity -------------------------------------------------------

def test_every_gold_answer_actually_appears_in_the_document():
    """Guards against gold rot: an expected string that isn't in the source
    document can never be retrieved, silently dragging the metrics down."""
    gold = ev.load_gold()
    text = (Path(dq._SAMPLES_DIR) / gold["document"]).read_text(encoding="utf-8").casefold()
    missing = [q["expected"] for q in gold["questions"] if q["expected"].casefold() not in text]
    assert not missing, f"gold answers not in {gold['document']}: {missing}"


def test_gold_questions_are_unique_and_categorised():
    gold = ev.load_gold()
    qs = [q["q"] for q in gold["questions"]]
    assert len(qs) == len(set(qs))
    assert {q["category"] for q in gold["questions"]} == {"lexical", "paraphrase"}


# --- the gate itself ----------------------------------------------------------

@pytest.mark.asyncio
async def test_current_retrieval_meets_recorded_minimums():
    report = await ev.run_eval("hashing", k=THRESHOLDS["k"])
    failures = ev.check_thresholds(report, THRESHOLDS)
    assert not failures, "retrieval regressed:\n" + "\n".join(failures)


@pytest.mark.asyncio
async def test_eval_is_deterministic():
    a = await ev.run_eval("hashing", k=4)
    b = await ev.run_eval("hashing", k=4)
    assert a == b


@pytest.mark.asyncio
async def test_gate_actually_fails_when_retrieval_is_broken(monkeypatch):
    """Mutation check: if search silently returned nothing, the gate must trip.
    Without this a gate that always passes would look identical to a good one."""
    from rag import SearchIndex

    monkeypatch.setattr(SearchIndex, "search", lambda self, *a, **kw: [])
    report = await ev.run_eval("hashing", k=THRESHOLDS["k"])
    assert report["modes"]["hybrid"]["overall"]["hit"] == 0.0
    assert ev.check_thresholds(report, THRESHOLDS), "gate did not fail on broken retrieval"


def test_check_thresholds_reports_each_failing_metric():
    report = {"k": 4, "modes": {"hybrid": {"overall": {"hit": 0.5, "mrr": 0.9}}}}
    thresholds = {"min": {"hybrid": {"overall": {"hit": 0.8, "mrr": 0.8}}}}
    failures = ev.check_thresholds(report, thresholds)
    assert len(failures) == 1 and "hit@4" in failures[0]


def test_cli_exits_nonzero_on_regression(tmp_path, monkeypatch):
    strict = {**THRESHOLDS, "min": {"hybrid": {"overall": {"hit": 1.01}}}}
    path = tmp_path / "strict.json"
    path.write_text(json.dumps(strict), encoding="utf-8")
    assert ev.main(["--check", "--thresholds", str(path)]) == 1


def test_cli_exits_two_when_no_thresholds_for_embedder(tmp_path):
    other = {**THRESHOLDS, "embedder": "minilm"}
    path = tmp_path / "other.json"
    path.write_text(json.dumps(other), encoding="utf-8")
    assert ev.main(["--check", "--thresholds", str(path)]) == 2
