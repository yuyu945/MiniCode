"""Runnable benchmark for code retrieval quality metrics."""

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
    print("Code Retrieval Benchmark")
    print("=" * 40)
    print(f"Workspace: {workspace}")
    print(f"Fixture:   {fixture_path}")
    for method, summary in metrics["summary"].items():
        print(f"[{method}]")
        print(f"  Top-1 File Hit Rate:   {summary['top1_file_hit_rate']:.2%}")
        print(f"  Top-5 File Hit Rate:   {summary['top5_file_hit_rate']:.2%}")
        print(f"  Top-5 Symbol Hit Rate: {summary['top5_symbol_hit_rate']:.2%}")
        print(f"  MRR:                   {summary['mrr']:.3f}")
        print(f"  Context Precision:     {summary['context_precision']:.3f}")
        print(f"  Avg Retrieved Chunks:  {summary['average_retrieved_chunks']:.2f}")
    print(f"Results saved to: {output_path}")


def main() -> int:
    repo_root = Path(__file__).parent.parent
    workspace = repo_root / "minicode"
    fixture_path = Path(__file__).parent / "fixtures" / "code_retrieval_cases.json"
    output_path = Path(__file__).parent / "code_retrieval_results.json"

    metrics = run_single_benchmark(workspace, fixture_path, output_path)
    print_benchmark_report(metrics, workspace, fixture_path, output_path)

    return 0 if metrics["summary"]["hybrid_rerank"]["top5_file_hit_rate"] >= 0.72 else 1


if __name__ == "__main__":
    raise SystemExit(main())
