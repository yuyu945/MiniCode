"""Runnable benchmark for bilingual code_intel quality."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from minicode.code_intel_benchmark import benchmark_code_intel


def run_single_benchmark(workspace: Path, fixture_path: Path, output_path: Path) -> dict:
    metrics = benchmark_code_intel(workspace, fixture_path)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def print_benchmark_report(metrics: dict, workspace: Path, fixture_path: Path, output_path: Path) -> None:
    print("Code Intel Quality Benchmark")
    print("=" * 40)
    print(f"Workspace: {workspace}")
    print(f"Fixture:   {fixture_path}")
    summary = metrics["summary"]
    print("[overall]")
    print(f"  Cases:     {summary['case_count']}")
    print(f"  Passed:    {summary['pass_count']}")
    print(f"  Pass Rate: {summary['pass_rate']:.2%}")
    for language, language_summary in sorted(metrics["language_summary"].items()):
        print(f"[language:{language}]")
        print(f"  Cases:     {language_summary['case_count']}")
        print(f"  Passed:    {language_summary['pass_count']}")
        print(f"  Pass Rate: {language_summary['pass_rate']:.2%}")
    for scenario, scenario_summary in sorted(metrics["scenario_summary"].items()):
        print(f"[scenario:{scenario}]")
        print(f"  Cases:     {scenario_summary['case_count']}")
        print(f"  Passed:    {scenario_summary['pass_count']}")
        print(f"  Pass Rate: {scenario_summary['pass_rate']:.2%}")
    for backend, backend_summary in sorted(metrics["backend_summary"].items()):
        print(f"[backend:{backend}]")
        print(f"  Cases:     {backend_summary['case_count']}")
        print(f"  Passed:    {backend_summary['pass_count']}")
        print(f"  Pass Rate: {backend_summary['pass_rate']:.2%}")
    print(f"Results saved to: {output_path}")


def main() -> int:
    repo_root = Path(__file__).parent.parent
    workspace = Path(__file__).parent / "fixtures" / "code_intel_quality_workspace"
    fixture_path = Path(__file__).parent / "fixtures" / "code_intel_quality_cases.json"
    output_path = Path(__file__).parent / "code_intel_quality_results.json"

    metrics = run_single_benchmark(workspace, fixture_path, output_path)
    print_benchmark_report(metrics, workspace, fixture_path, output_path)

    return 0 if metrics["summary"]["pass_rate"] >= 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
