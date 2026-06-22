"""Cross-repo benchmark runner for code retrieval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.code_retrieval_benchmark import print_benchmark_report, run_single_benchmark


def _aggregate_repo_summaries(repo_results: dict[str, dict]) -> dict[str, dict]:
    methods = ["baseline_dense", "hybrid", "hybrid_rerank"]
    aggregate: dict[str, dict] = {}
    for method in methods:
        summaries = [result["summary"][method] for result in repo_results.values()]
        total = max(len(summaries), 1)
        aggregate[method] = {
            "top1_file_hit_rate": round(sum(s["top1_file_hit_rate"] for s in summaries) / total, 4),
            "top5_file_hit_rate": round(sum(s["top5_file_hit_rate"] for s in summaries) / total, 4),
            "top5_symbol_hit_rate": round(sum(s["top5_symbol_hit_rate"] for s in summaries) / total, 4),
            "mrr": round(sum(s["mrr"] for s in summaries) / total, 4),
            "context_precision": round(sum(s["context_precision"] for s in summaries) / total, 4),
            "average_retrieved_chunks": round(sum(s["average_retrieved_chunks"] for s in summaries) / total, 4),
        }
    return aggregate


def main() -> int:
    repo_root = Path(__file__).parent.parent
    suite = {
        "minicode_python": {
            "workspace": repo_root / "minicode",
            "fixture": Path(__file__).parent / "fixtures" / "code_retrieval_cases.json",
            "output": Path(__file__).parent / "code_retrieval_results.json",
        },
        "codec": {
            "workspace": Path("D:/Python/agent/CodeC/src"),
            "fixture": Path(__file__).parent / "fixtures" / "codec_code_retrieval_cases.json",
            "output": Path(__file__).parent / "codec_code_retrieval_results.json",
        },
        "super_agent_gopy_python": {
            "workspace": Path("D:/Python/agent/super-agent-gopy/python/app"),
            "fixture": Path(__file__).parent / "fixtures" / "super_agent_gopy_code_retrieval_cases.json",
            "output": Path(__file__).parent / "super_agent_gopy_code_retrieval_results.json",
        },
        "claude_code_rev": {
            "workspace": Path("D:/Python/agent/cc/claude-code-rev/src"),
            "fixture": Path(__file__).parent / "fixtures" / "claude_code_rev_cases.json",
            "output": Path(__file__).parent / "claude_code_rev_code_retrieval_results.json",
        },
    }

    repo_results: dict[str, dict] = {}
    for name, config in suite.items():
        print(f"\n=== Repo: {name} ===")
        metrics = run_single_benchmark(config["workspace"], config["fixture"], config["output"])
        print_benchmark_report(metrics, config["workspace"], config["fixture"], config["output"])
        repo_results[name] = metrics

    aggregate = _aggregate_repo_summaries(repo_results)
    output_path = Path(__file__).parent / "multi_repo_code_retrieval_results.json"
    payload = {
        "repos": repo_results,
        "aggregate_summary": aggregate,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Aggregate Summary ===")
    for method, summary in aggregate.items():
        print(f"[{method}] Top-5 File Hit Rate: {summary['top5_file_hit_rate']:.2%}, "
              f"Top-5 Symbol Hit Rate: {summary['top5_symbol_hit_rate']:.2%}, "
              f"MRR: {summary['mrr']:.3f}")
    print(f"Results saved to: {output_path}")

    return 0 if aggregate["hybrid_rerank"]["top5_file_hit_rate"] >= 0.60 else 1


if __name__ == "__main__":
    raise SystemExit(main())
