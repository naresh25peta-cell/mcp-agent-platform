"""
retrieval_eval.py — Phase 4: score the document-qa MCP server's retrieval and
fail the build if it regresses.

Every question is sent through the real ``search_documents`` MCP tool (not by
calling the retrieval library directly), so the check covers chunking, the
FAISS + BM25Plus fusion, and the MCP tool layer together.

Metrics (retrieval only; this is NOT answer accuracy):
  - hit@k : share of questions where any of the top-k chunks contains the
            expected text
  - MRR@k : mean of 1/rank of the first matching chunk (0 if none in top-k)

Usage:
    python -m mcp_agent_platform.evals.retrieval_eval                 # print table
    python -m mcp_agent_platform.evals.retrieval_eval --check         # CI gate
    python -m mcp_agent_platform.evals.retrieval_eval --embedder minilm
    python -m mcp_agent_platform.evals.retrieval_eval --report out.json

Exit code is 1 when ``--check`` finds a metric below its threshold.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp_agent_platform.evals.embedders import get_embedder

EVALS_DIR = Path(__file__).resolve().parent
MODES = ("keyword", "semantic", "hybrid")


def load_gold(path: Path | None = None) -> dict:
    return json.loads((path or EVALS_DIR / "gold_retrieval.json").read_text(encoding="utf-8"))


def first_hit_rank(texts: list[str], expected: str) -> int | None:
    """1-based rank of the first text containing `expected` (case-insensitive)."""
    needle = expected.casefold()
    for rank, text in enumerate(texts, start=1):
        if needle in text.casefold():
            return rank
    return None


def score(ranks: list[int | None]) -> dict:
    """hit@k and MRR@k from a list of first-hit ranks (None = miss)."""
    n = len(ranks)
    if n == 0:
        return {"n": 0, "hit": 0.0, "mrr": 0.0}
    hits = sum(1 for r in ranks if r is not None)
    mrr = sum(1 / r for r in ranks if r is not None)
    return {"n": n, "hit": hits / n, "mrr": mrr / n}


async def run_eval(embedder_name: str = "hashing", k: int = 4, gold: dict | None = None) -> dict:
    """Run every gold question through the MCP tool in every mode."""
    from mcp_agent_platform.servers import document_qa_server as dq

    gold = gold or load_gold()
    embedder = get_embedder(embedder_name)
    original_get_model = dq._get_model
    dq._get_model = lambda: embedder
    dq._indexes.clear()
    try:
        ranks: dict[str, list[tuple[str, int | None]]] = {m: [] for m in MODES}
        for mode in MODES:
            for item in gold["questions"]:
                result = await dq.mcp.call_tool(
                    "search_documents",
                    {"question": item["q"], "document": gold["document"], "top_k": k, "mode": mode},
                )
                if result.is_error:
                    raise RuntimeError(f"search_documents failed for {item['q']!r}")
                texts = [c["text"] for c in result.structured_content["result"]]
                ranks[mode].append((item["category"], first_hit_rank(texts, item["expected"])))
    finally:
        dq._get_model = original_get_model
        dq._indexes.clear()

    report: dict = {"embedder": embedder_name, "k": k, "questions": len(gold["questions"]), "modes": {}}
    for mode in MODES:
        entry = {"overall": score([r for _, r in ranks[mode]])}
        for category in sorted({c for c, _ in ranks[mode]}):
            entry[category] = score([r for c, r in ranks[mode] if c == category])
        report["modes"][mode] = entry
    return report


def check_thresholds(report: dict, thresholds: dict) -> list[str]:
    """Return a human-readable failure for every metric below its minimum."""
    failures = []
    for mode, groups in thresholds.get("min", {}).items():
        for group, minimums in groups.items():
            actual = report["modes"][mode][group]
            for metric, minimum in minimums.items():
                if actual[metric] < minimum:
                    failures.append(
                        f"{mode}/{group} {metric}@{report['k']} = {actual[metric]:.3f} "
                        f"< minimum {minimum:.3f}"
                    )
    return failures


def format_table(report: dict) -> str:
    k = report["k"]
    lines = [
        f"Retrieval eval — embedder={report['embedder']}, {report['questions']} questions, k={k}",
        "(retrieval only; not answer accuracy)",
        f"{'mode':10} {'group':11} {'n':>3} {'hit@' + str(k):>7} {'MRR@' + str(k):>7}",
    ]
    for mode, groups in report["modes"].items():
        for group, s in groups.items():
            lines.append(f"{mode:10} {group:11} {s['n']:>3} {s['hit']:>7.3f} {s['mrr']:>7.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--embedder", choices=("hashing", "minilm"), default="hashing")
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("--check", action="store_true", help="fail (exit 1) if below thresholds")
    parser.add_argument("--thresholds", type=Path, default=EVALS_DIR / "thresholds.json")
    parser.add_argument("--report", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)

    report = asyncio.run(run_eval(args.embedder, args.k))
    print(format_table(report))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.check:
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
        if thresholds.get("embedder") != args.embedder or thresholds.get("k") != args.k:
            print(
                f"\nNo thresholds recorded for embedder={args.embedder}, k={args.k} "
                f"(file is for embedder={thresholds.get('embedder')}, k={thresholds.get('k')}).",
                file=sys.stderr,
            )
            return 2
        failures = check_thresholds(report, thresholds)
        if failures:
            print("\nREGRESSION — retrieval fell below the recorded minimums:", file=sys.stderr)
            for f in failures:
                print(f"  - {f}", file=sys.stderr)
            return 1
        print("\nOK — all metrics at or above the recorded minimums.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
