"""Staged code retrieval pipeline for local workspace source search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minicode.retrieval.code_index import CodeIndex
from minicode.retrieval.code_pipeline import CodeRetrievalPipeline, evidence_to_result_dict
from minicode.retrieval.types import CodeRetrievalResult


class CodeRetrieval:
    """Workspace code retrieval as a staged control loop."""

    def __init__(self) -> None:
        self._index = CodeIndex()
        self._pipeline = CodeRetrievalPipeline(self._index)
        self._workspace: Path | None = None

    def index_workspace(self, workspace_path: str | Path) -> "CodeRetrieval":
        self._workspace = Path(workspace_path)
        self._index.build(workspace_path)
        return self

    def benchmark_ready_index(self, workspace_path: str | Path) -> "CodeRetrieval":
        return self.index_workspace(workspace_path)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "indexed_files": len(self._index.import_graph),
            "indexed_chunks": len(self._index.chunks),
            "languages": sorted({chunk.language for chunk in self._index.chunks}),
            "failed_files": list(self._index.failed_files),
        }

    def retrieve(self, query: str, top_k: int = 5, dependency_hops: int = 1) -> CodeRetrievalResult:
        return self._pipeline.retrieve(query, top_k=top_k, dependency_hops=dependency_hops)

    def search(
        self,
        query: str,
        top_k: int = 5,
        language_scope: list[str] | None = None,
        symbol_kinds: list[str] | None = None,
        *,
        mode: str = "hybrid_rerank",
    ) -> list[dict[str, Any]]:
        del mode  # legacy compatibility; staged retrieval no longer switches algorithms
        result = self.retrieve(query, top_k=max(top_k * 2, top_k))
        chunk_lookup = {
            (chunk.path, chunk.symbol_name, chunk.start_line): chunk
            for chunk in self._index.chunks
        }
        items: list[dict[str, Any]] = []
        for evidence in result.candidates:
            chunk = chunk_lookup.get((evidence.path, evidence.symbol_name, evidence.start_line))
            if chunk is None:
                continue
            if language_scope and chunk.language not in set(language_scope):
                continue
            if symbol_kinds and evidence.symbol_kind not in set(symbol_kinds):
                continue
            items.append(evidence_to_result_dict(evidence, chunk.language, chunk.chunk_id, chunk.signature))
        return items[:top_k]


def benchmark_code_retrieval(
    workspace_path: str | Path,
    fixture_path: str | Path,
    top_k: int = 5,
) -> dict[str, Any]:
    retrieval = CodeRetrieval().benchmark_ready_index(workspace_path)
    cases = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    methods = ["baseline_dense", "hybrid", "hybrid_rerank"]
    per_method_cases: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    merged_cases: list[dict[str, Any]] = []

    for case in cases:
        query = case["query"]
        expected_targets = case["expected_targets"]
        case_record = {"query": query, "expected_targets": expected_targets, "methods": {}}
        for method in methods:
            results = retrieval.search(query, top_k=top_k, mode=method)
            file_hit_rank = _find_hit_rank(results, expected_targets, symbol_only=False)
            symbol_hit_rank = _find_hit_rank(results, expected_targets, symbol_only=True)
            relevant_count = sum(1 for result in results if _matches_expected(result, expected_targets))
            context_precision = relevant_count / len(results) if results else 0.0
            method_case = {
                "query": query,
                "file_hit_rank": file_hit_rank,
                "symbol_hit_rank": symbol_hit_rank,
                "top_result": results[0] if results else None,
                "results": results,
                "expected_targets": expected_targets,
                "context_precision": round(context_precision, 4),
                "retrieved_chunks": len(results),
            }
            per_method_cases[method].append(method_case)
            case_record["methods"][method] = method_case
        merged_cases.append(case_record)

    summaries = {
        method: _summarize_method(per_method_cases[method], top_k=top_k)
        for method in methods
    }
    return {
        "workspace": str(workspace_path),
        "fixture_path": str(fixture_path),
        "methods": per_method_cases,
        "summary": summaries,
        "cases": merged_cases,
        "stats": retrieval.stats,
    }


def _matches_expected(
    result: dict[str, Any],
    expected_targets: list[dict[str, Any]],
    *,
    symbol_only: bool = False,
) -> bool:
    result_path = result["path"].lower()
    result_name = result["symbol_name"].lower()
    result_kind = result["symbol_kind"].lower()
    for expected in expected_targets:
        path_match = expected.get("path")
        name_match = expected.get("symbol_name")
        kind_match = expected.get("symbol_kind")
        if path_match and Path(path_match).as_posix().lower() != result_path:
            continue
        if symbol_only and not (name_match or kind_match):
            continue
        if name_match and str(name_match).lower() != result_name:
            continue
        if kind_match and str(kind_match).lower() != result_kind:
            continue
        return True
    return False


def _find_hit_rank(
    results: list[dict[str, Any]],
    expected_targets: list[dict[str, Any]],
    *,
    symbol_only: bool,
) -> int | None:
    for index, result in enumerate(results, start=1):
        if _matches_expected(result, expected_targets, symbol_only=symbol_only):
            return index
    return None


def _summarize_method(cases: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    total = max(len(cases), 1)
    top1_file_hits = sum(1 for case in cases if case["file_hit_rank"] == 1)
    top5_file_hits = sum(1 for case in cases if case["file_hit_rank"] is not None and case["file_hit_rank"] <= top_k)
    top5_symbol_hits = sum(1 for case in cases if case["symbol_hit_rank"] is not None and case["symbol_hit_rank"] <= top_k)
    reciprocal_ranks = [1.0 / case["file_hit_rank"] if case["file_hit_rank"] is not None else 0.0 for case in cases]
    avg_precision = sum(case["context_precision"] for case in cases) / total
    avg_chunks = sum(case["retrieved_chunks"] for case in cases) / total
    return {
        "top1_file_hit_rate": round(top1_file_hits / total, 4),
        "top5_file_hit_rate": round(top5_file_hits / total, 4),
        "top5_symbol_hit_rate": round(top5_symbol_hits / total, 4),
        "mrr": round(sum(reciprocal_ranks) / total, 4),
        "context_precision": round(avg_precision, 4),
        "average_retrieved_chunks": round(avg_chunks, 4),
    }
