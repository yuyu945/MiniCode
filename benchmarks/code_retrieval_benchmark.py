"""Runnable benchmark for the staged code retrieval pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from minicode.code_retrieval import benchmark_code_retrieval


def run_single_benchmark(
    workspace: Path,
    fixture_path: Path,
    output_path: Path,
) -> dict:
    metrics = benchmark_code_retrieval(workspace, fixture_path)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics


def print_benchmark_report(metrics: dict, workspace: Path, fixture_path: Path, output_path: Path) -> None:
    summary = metrics["summary"]
    print("Code Retrieval Benchmark")
    print("=" * 40)
    print(f"Workspace: {workspace}")
    print(f"Fixture:   {fixture_path}")
    print("[overall]")
    print(f"  Cases:               {summary['case_count']}")
    print(f"  Top-1 File Hit Rate: {summary['top1_file_hit_rate']:.2%}")
    print(f"  Top-{5} File Hit Rate: {summary['top5_file_hit_rate']:.2%}")
    print(f"  Top-{5} Symbol Hit Rate: {summary['top5_symbol_hit_rate']:.2%}")
    print(f"  MRR:                 {summary['mrr']:.3f}")
    print(f"  Context Precision:   {summary['context_precision']:.3f}")
    print(f"  Avg Retrieved:       {summary['average_retrieved_chunks']:.2f}")
    for query_type, query_summary in sorted(metrics["query_type_summary"].items()):
        print(f"[type:{query_type}]")
        print(f"  Cases:               {query_summary['case_count']}")
        print(f"  Top-1 File Hit Rate: {query_summary['top1_file_hit_rate']:.2%}")
        print(f"  Top-5 File Hit Rate: {query_summary['top5_file_hit_rate']:.2%}")
        print(f"  MRR:                 {query_summary['mrr']:.3f}")
    print(f"Results saved to: {output_path}")


def main() -> int:
    repo_root = Path(__file__).parent.parent
    workspace = repo_root / "minicode"
    fixture_path = Path(__file__).parent / "fixtures" / "staged_code_retrieval_cases.json"
    output_path = Path(__file__).parent / "code_retrieval_results.json"

    metrics = run_single_benchmark(workspace, fixture_path, output_path)
    print_benchmark_report(metrics, workspace, fixture_path, output_path)

    return 0 if metrics["summary"]["top5_file_hit_rate"] >= 0.65 else 1


if __name__ == "__main__":
    raise SystemExit(main())
