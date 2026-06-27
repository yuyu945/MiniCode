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
        passed = all(fragment in result.output for fragment in expected_substrings)
        case_results.append(
            {
                "case_id": case.get("case_id"),
                "language": str(case.get("language", "unknown")),
                "operation": operation,
                "backend": result.backend,
                "passed": passed,
                "expected_substrings": expected_substrings,
                "output": result.output,
            }
        )

    return {
        "workspace": str(workspace),
        "fixture_path": str(fixture_path),
        "summary": _summarize(case_results),
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


def _summarize(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for case in case_results if case["passed"])
    return {
        "case_count": total,
        "pass_count": passed,
        "pass_rate": round((passed / total), 4) if total else 0.0,
    }
