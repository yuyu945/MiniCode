"""Tests for code retrieval indexing, search, and benchmark metrics."""

from __future__ import annotations

import json
from pathlib import Path

from minicode.code_retrieval import CodeRetrieval, benchmark_code_retrieval
from minicode.retrieval.code_index import CodeIndex
from minicode.retrieval.types import CodeRetrievalResult, RetrievalIntent


def _write_repo_query_fixture(workspace: Path) -> Path:
    workspace.mkdir()
    (workspace / "agent_loop.py").write_text(
        "def run_agent_turn(task: str) -> str:\n"
        "    return task\n",
        encoding="utf-8",
    )
    (workspace / "context_compactor.py").write_text(
        "def _run_full_compact(turn: str) -> str:\n"
        "    return turn\n",
        encoding="utf-8",
    )
    (workspace / "cybernetic_orchestrator.py").write_text(
        "def run_cost_control(turn: str) -> str:\n"
        "    return turn\n",
        encoding="utf-8",
    )
    return workspace


def _write_permission_fixture(workspace: Path) -> Path:
    workspace.mkdir()
    (workspace / "permissions.py").write_text(
        "class PermissionManager:\n"
        "    def decide(self, action: str) -> bool:\n"
        "        return bool(action)\n"
        "\n"
        "class PermissionGate:\n"
        "    def check(self, action: str) -> bool:\n"
        "        return bool(action)\n",
        encoding="utf-8",
    )
    (workspace / "auto_mode.py").write_text(
        "class PermissionMode:\n"
        "    pass\n",
        encoding="utf-8",
    )
    return workspace


def _write_llm_fixture(workspace: Path) -> Path:
    (workspace / "infra").mkdir(parents=True)
    (workspace / "infra" / "llm.py").write_text(
        "class LLMClient:\n"
        "    def complete(self, prompt: str) -> str:\n"
        "        return prompt\n",
        encoding="utf-8",
    )
    (workspace / "infra" / "llm_observability.py").write_text(
        "class ObservableLLMClient:\n"
        "    def record(self, prompt: str) -> str:\n"
        "        return prompt\n",
        encoding="utf-8",
    )
    return workspace


def test_code_retrieval_result_supports_multi_stage_evidence():
    result = CodeRetrievalResult(
        query="where does login validation happen",
        intent=RetrievalIntent(
            query="where does login validation happen",
            symbols=["login", "validate"],
            keywords=["login", "validation"],
            file_hints=["auth", "user"],
            stage_budget=5,
            dependency_hops=1,
        ),
        candidates=[],
        expansions=[],
        corrections=[],
    )

    assert result.intent.stage_budget == 5
    assert result.expansions == []
    assert result.corrections == []


def test_code_retrieval_indexes_python_and_typescript_symbols(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "auth.py").write_text(
        "class AuthService:\n"
        "    def login(self, token: str) -> bool:\n"
        "        return token == 'ok'\n",
        encoding="utf-8",
    )
    (workspace / "session.ts").write_text(
        "export class SessionStore {\n"
        "  getSession(id: string) { return id; }\n"
        "}\n"
        "export function createSession(userId: string) { return userId; }\n",
        encoding="utf-8",
    )

    retrieval = CodeRetrieval().index_workspace(workspace)
    stats = retrieval.stats

    assert stats["indexed_files"] == 2
    assert stats["indexed_chunks"] >= 4
    assert stats["indexed_definitions"] >= 3
    assert {"python", "typescript"} <= set(stats["languages"])


def test_code_index_tracks_symbol_definitions_and_references(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "auth.py").write_text(
        "from validator import validate_user\n"
        "def login(token: str):\n"
        "    return validate_user(token)\n",
        encoding="utf-8",
    )
    (workspace / "validator.py").write_text(
        "def validate_user(token: str):\n"
        "    return token == 'ok'\n",
        encoding="utf-8",
    )

    index = CodeIndex().build(workspace)

    definitions = index.definition_chunks("validate_user")
    references = index.reference_entries("validate_user")

    assert any(item.path == "validator.py" for item in definitions)
    assert any(item.path == "auth.py" and item.source_symbol == "login" for item in references)


def test_code_retrieval_search_returns_structured_symbol_results(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "auth.py").write_text(
        "class AuthService:\n"
        "    def login(self, token: str) -> bool:\n"
        "        return token == 'ok'\n",
        encoding="utf-8",
    )

    retrieval = CodeRetrieval().index_workspace(workspace)
    results = retrieval.search("login token auth service", top_k=5)

    assert results
    top = results[0]
    assert {
        "chunk_id",
        "path",
        "language",
        "symbol_name",
        "symbol_kind",
        "score",
        "matched_terms",
    } <= set(top)
    assert top["symbol_name"] in {"AuthService", "login"}


def test_code_retrieval_expands_from_login_to_validator(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "auth.py").write_text(
        "from validator import validate_user\n"
        "def login(token: str):\n"
        "    return validate_user(token)\n",
        encoding="utf-8",
    )
    (workspace / "validator.py").write_text(
        "def validate_user(token: str):\n"
        "    return token == 'ok'\n",
        encoding="utf-8",
    )

    retrieval = CodeRetrieval().index_workspace(workspace)
    results = retrieval.search("where is login validation implemented", top_k=5)

    paths = {item["path"] for item in results}
    assert "auth.py" in paths
    assert "validator.py" in paths


def test_code_retrieval_records_correction_when_first_guess_is_weak(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "session.py").write_text(
        "def create_session(user_id: str):\n"
        "    return user_id\n",
        encoding="utf-8",
    )

    retrieval = CodeRetrieval().index_workspace(workspace)
    result = retrieval.retrieve("find login code", top_k=3)

    assert isinstance(result.corrections, list)


def test_code_retrieval_benchmark_reports_topk_metrics(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "memory.py").write_text(
        "class MemoryManager:\n"
        "    def search(self, query: str):\n"
        "        return [query]\n",
        encoding="utf-8",
    )
    (workspace / "session.ts").write_text(
        "export function loadSession(id: string) { return id; }\n",
        encoding="utf-8",
    )
    fixture_path = tmp_path / "cases.json"
    fixture_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "case-1",
                    "query_type": "cross_file_dependency",
                    "query": "where is memory search implemented",
                    "expected_targets": [
                        {"path": "memory.py", "symbol_name": "search", "symbol_kind": "method"},
                    ],
                },
                {
                    "query_id": "case-2",
                    "query_type": "direct_symbol_lookup",
                    "query": "find session loader",
                    "expected_targets": [
                        {"path": "session.ts", "symbol_name": "loadSession", "symbol_kind": "function"},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    metrics = benchmark_code_retrieval(workspace, fixture_path)

    assert {"summary", "query_type_summary", "cases"} <= set(metrics)
    assert {
        "top1_file_hit_rate",
        "top5_file_hit_rate",
        "top5_symbol_hit_rate",
        "mrr",
        "context_precision",
        "average_retrieved_chunks",
        "case_count",
    } <= set(metrics["summary"])
    assert {"cross_file_dependency", "direct_symbol_lookup"} <= set(metrics["query_type_summary"])
    assert len(metrics["cases"]) == 2
    assert 0.0 <= metrics["summary"]["top5_file_hit_rate"] <= 1.0


def test_code_retrieval_prefers_real_runtime_entrypoint_on_repo_queries(tmp_path):
    workspace = _write_repo_query_fixture(tmp_path / "repo")
    retrieval = CodeRetrieval().index_workspace(workspace)

    results = retrieval.search("where does the agent run a full turn", top_k=5, mode="hybrid_rerank")

    assert results
    assert results[0]["path"] == "agent_loop.py"
    assert results[0]["symbol_name"] == "run_agent_turn"


def test_code_retrieval_prefers_manager_class_for_permission_query(tmp_path):
    workspace = _write_permission_fixture(tmp_path / "repo")
    retrieval = CodeRetrieval().index_workspace(workspace)

    results = retrieval.search("which class owns permission decisions", top_k=5, mode="hybrid_rerank")

    assert results
    assert results[0]["path"] == "permissions.py"
    assert results[0]["symbol_name"] == "PermissionManager"


def test_code_retrieval_splits_acronym_identifiers_for_class_lookup(tmp_path):
    workspace = _write_llm_fixture(tmp_path / "repo")
    retrieval = CodeRetrieval().index_workspace(workspace)

    results = retrieval.search("find the llm client", top_k=5, mode="hybrid_rerank")

    assert results
    assert results[0]["path"] == "infra/llm.py"
    assert results[0]["symbol_name"] == "LLMClient"
