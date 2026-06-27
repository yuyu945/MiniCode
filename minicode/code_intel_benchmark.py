"""Benchmark helpers for code_intel quality across languages and operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minicode.code_intel_backend import select_code_intel_backend


def benchmark_code_intel(
    workspace_path: str | Path,
    fixture_path: str | Path,
) -> dict[str, Any]:
    workspace = Path(workspace_path)
    cases = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    case_results: list[dict[str, Any]] = []

    for case in cases:
        operation = str(case["operation"])
        symbol = case.get("symbol")
        file_path = case.get("file_path")
        backend = select_code_intel_backend(workspace, file_path or _guess_file_path(workspace, str(case.get("language", ""))))
        result = backend.run(operation, symbol=symbol, file_path=file_path)
        expected_substrings = [str(item) for item in case.get("expected_substrings", [])]
        expected_ordered = [str(item) for item in case.get("expected_ordered_substrings", [])]
        unexpected_substrings = [str(item) for item in case.get("unexpected_substrings", [])]
        passed = _matches_expectations(
            result.output,
            expected_substrings=expected_substrings,
            expected_ordered=expected_ordered,
            unexpected_substrings=unexpected_substrings,
        )
        case_results.append(
            {
                "case_id": case.get("case_id"),
                "scenario_type": str(case.get("scenario_type", "general")),
                "language": str(case.get("language", "unknown")),
                "operation": operation,
                "backend": result.backend,
                "passed": passed,
                "expected_substrings": expected_substrings,
                "expected_ordered_substrings": expected_ordered,
                "unexpected_substrings": unexpected_substrings,
                "assertion_counts": {
                    "contains": len(expected_substrings),
                    "ordered": len(expected_ordered),
                    "unexpected": len(unexpected_substrings),
                },
                "output": result.output,
            }
        )

    return {
        "workspace": str(workspace),
        "fixture_path": str(fixture_path),
        "summary": _summarize(case_results),
        "scenario_summary": _group_summary(case_results, "scenario_type"),
        "language_summary": _group_summary(case_results, "language"),
        "operation_summary": _group_summary(case_results, "operation"),
        "backend_summary": _group_summary(case_results, "backend"),
        "cases": case_results,
    }


def _guess_file_path(workspace: Path, language: str) -> str | None:
    patterns = {
        "python": ("*.py",),
        "typescript": ("*.ts", "*.tsx"),
    }
    for pattern in patterns.get(language.lower(), ()):
        match = next(workspace.rglob(pattern), None)
        if match is not None:
            return match.relative_to(workspace).as_posix()
    return None


def _group_summary(case_results: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    values = sorted({str(case.get(key, "unknown")) for case in case_results})
    return {
        value: _summarize([case for case in case_results if str(case.get(key, "unknown")) == value])
        for value in values
    }


def _matches_expectations(
    output: str,
    *,
    expected_substrings: list[str],
    expected_ordered: list[str],
    unexpected_substrings: list[str],
) -> bool:
    if not all(fragment in output for fragment in expected_substrings):
        return False
    cursor = 0
    for fragment in expected_ordered:
        idx = output.find(fragment, cursor)
        if idx < 0:
            return False
        cursor = idx + len(fragment)
    if any(fragment in output for fragment in unexpected_substrings):
        return False
    return True


def _summarize(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for case in case_results if case["passed"])
    return {
        "case_count": total,
        "pass_count": passed,
        "pass_rate": round((passed / total), 4) if total else 0.0,
    }
