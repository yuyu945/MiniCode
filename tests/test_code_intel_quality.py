from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from minicode.code_intel_backend import IndexCodeIntelBackend
from minicode.code_intel_benchmark import benchmark_code_intel
from minicode.tools.code_intel import code_intel_tool
from minicode.tooling import ToolContext


def test_typescript_index_backend_tracks_camel_case_references(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "service.ts").write_text(
        "export function loadUser(id: string) {\n"
        "  return { id };\n"
        "}\n"
        "\n"
        "export function buildSession(userId: string) {\n"
        "  return loadUser(userId);\n"
        "}\n",
        encoding="utf-8",
    )

    backend = IndexCodeIntelBackend(workspace)
    result = backend.run("find_references", symbol="loadUser")

    assert result.ok is True
    assert "service.ts:6" in result.output
    assert "buildSession" in result.output


def test_typescript_index_backend_document_symbols_include_type_aliases_in_tsx(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "dashboard.tsx").write_text(
        "type DashboardProps = {\n"
        "  title: string;\n"
        "};\n"
        "\n"
        "function useDashboardState() {\n"
        "  return { ready: true };\n"
        "}\n"
        "\n"
        "export function DashboardPanel(props: DashboardProps) {\n"
        "  const state = useDashboardState();\n"
        "  return <section>{props.title}:{String(state.ready)}</section>;\n"
        "}\n",
        encoding="utf-8",
    )

    backend = IndexCodeIntelBackend(workspace)
    result = backend.run("document_symbols", file_path="dashboard.tsx")

    assert result.ok is True
    assert "DashboardPanel" in result.output
    assert "useDashboardState" in result.output
    assert "DashboardProps" in result.output


def test_typescript_external_lsp_opens_related_files_for_cross_file_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "service.ts").write_text(
        "export function loadUser(id: string) {\n"
        "  return { id };\n"
        "}\n",
        encoding="utf-8",
    )
    (workspace / "consumer.ts").write_text(
        "import { loadUser } from \"./service\";\n"
        "\n"
        "export function consume(id: string) {\n"
        "  return loadUser(id);\n"
        "}\n",
        encoding="utf-8",
    )

    server_script = tmp_path / "fake_ts_quality_lsp.py"
    server_script.write_text(
        """
import json
import sys
from urllib.parse import quote

opened = set()

def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\\r\\n", b"\\n"):
            break
        key, value = line.decode("utf-8").split(":", 1)
        headers[key.strip().lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode("utf-8"))

def send_message(payload):
    body = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\\r\\n\\r\\n".encode("utf-8"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()

def to_uri(path):
    normalized = path.replace("\\\\", "/")
    if ":" in normalized[:3]:
        return "file:///" + quote(normalized, safe="/:")
    return "file://" + quote(normalized, safe="/:")

service_uri = to_uri(sys.argv[1])
consumer_uri = to_uri(sys.argv[2])

while True:
    msg = read_message()
    if msg is None:
        break
    method = msg.get("method")
    if method == "initialize":
        send_message({"jsonrpc": "2.0", "id": msg["id"], "result": {"capabilities": {}}})
    elif method == "initialized":
        continue
    elif method == "textDocument/didOpen":
        opened.add(msg["params"]["textDocument"]["uri"])
    elif method == "workspace/symbol":
        send_message({
            "jsonrpc": "2.0",
            "id": msg["id"],
            "result": [{
                "name": "loadUser",
                "kind": 12,
                "location": {
                    "uri": service_uri,
                    "range": {
                        "start": {"line": 0, "character": 16},
                        "end": {"line": 0, "character": 24}
                    }
                }
            }]
        })
    elif method == "textDocument/references":
        result = [{
            "uri": service_uri,
            "range": {
                "start": {"line": 0, "character": 16},
                "end": {"line": 0, "character": 24}
            }
        }]
        if service_uri in opened and consumer_uri in opened:
            result.append({
                "uri": consumer_uri,
                "range": {
                    "start": {"line": 3, "character": 9},
                    "end": {"line": 3, "character": 17}
                }
            })
        send_message({"jsonrpc": "2.0", "id": msg["id"], "result": result})
    elif method in ("textDocument/definition", "textDocument/implementation"):
        send_message({
            "jsonrpc": "2.0",
            "id": msg["id"],
            "result": [{
                "uri": service_uri,
                "range": {
                    "start": {"line": 0, "character": 16},
                    "end": {"line": 0, "character": 24}
                }
            }]
        })
    elif method == "textDocument/hover":
        send_message({
            "jsonrpc": "2.0",
            "id": msg["id"],
            "result": {"contents": "function loadUser(id: string)"}
        })
    elif method == "textDocument/documentSymbol":
        send_message({
            "jsonrpc": "2.0",
            "id": msg["id"],
            "result": [{
                "name": "loadUser",
                "kind": 12,
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 2, "character": 1}
                },
                "selectionRange": {
                    "start": {"line": 0, "character": 16},
                    "end": {"line": 0, "character": 24}
                }
            }]
        })
    elif method == "shutdown":
        send_message({"jsonrpc": "2.0", "id": msg["id"], "result": None})
        break
    elif method == "exit":
        break
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "MINICODE_TYPESCRIPT_LSP_COMMAND",
        json.dumps(
            [
                sys.executable,
                str(server_script),
                str((workspace / "service.ts").resolve()),
                str((workspace / "consumer.ts").resolve()),
            ]
        ),
    )

    result = code_intel_tool.run(
        {"operation": "find_references", "symbol": "loadUser", "path": "."},
        ToolContext(cwd=str(workspace), permissions=None),
    )

    assert result.ok is True
    assert "Backend: typescript_external_lsp" in result.output
    assert "consumer.ts:4-4" in result.output


def test_code_intel_benchmark_reports_language_and_backend_quality(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "auth.py").write_text(
        "def normalize_email(value: str) -> str:\n"
        "    return value.strip().lower()\n",
        encoding="utf-8",
    )
    (workspace / "service.ts").write_text(
        "export function loadUser(id: string) {\n"
        "  return { id };\n"
        "}\n"
        "\n"
        "export function buildSession(userId: string) {\n"
        "  return loadUser(userId);\n"
        "}\n",
        encoding="utf-8",
    )

    fixture_path = tmp_path / "code_intel_cases.json"
    fixture_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "py-definition",
                    "language": "python",
                    "operation": "go_to_definition",
                    "symbol": "normalize_email",
                    "expected_substrings": ["auth.py"],
                },
                {
                    "case_id": "ts-references",
                    "language": "typescript",
                    "operation": "find_references",
                    "symbol": "loadUser",
                    "expected_substrings": ["service.ts:6"],
                },
            ]
        ),
        encoding="utf-8",
    )

    metrics = benchmark_code_intel(workspace, fixture_path)

    assert {"summary", "language_summary", "operation_summary", "cases"} <= set(metrics)
    assert metrics["summary"]["case_count"] == 2
    assert {"python", "typescript"} <= set(metrics["language_summary"])
    assert {"go_to_definition", "find_references"} <= set(metrics["operation_summary"])
    assert 0.0 <= metrics["summary"]["pass_rate"] <= 1.0


def test_code_intel_benchmark_groups_cases_by_scenario_type(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "service.py").write_text(
        "class UserService:\n"
        "    def normalize_email(self, value: str) -> str:\n"
        "        return value.strip().lower()\n",
        encoding="utf-8",
    )
    fixture_path = tmp_path / "code_intel_cases.json"
    fixture_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "workspace-ambiguity",
                    "scenario_type": "workspace_symbol_ambiguity",
                    "language": "python",
                    "operation": "workspace_symbol",
                    "symbol": "UserService",
                    "expected_substrings": ["UserService"],
                },
                {
                    "case_id": "method-definition",
                    "scenario_type": "multiple_definitions",
                    "language": "python",
                    "operation": "go_to_definition",
                    "symbol": "normalize_email",
                    "expected_substrings": ["service.py"],
                },
            ]
        ),
        encoding="utf-8",
    )

    metrics = benchmark_code_intel(workspace, fixture_path)

    assert "scenario_summary" in metrics
    assert {"workspace_symbol_ambiguity", "multiple_definitions"} <= set(metrics["scenario_summary"])
    assert metrics["scenario_summary"]["workspace_symbol_ambiguity"]["case_count"] == 1


def test_code_intel_benchmark_supports_absent_and_ordered_expectations(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "alpha.py").write_text("class SearchService:\n    pass\n", encoding="utf-8")
    (workspace / "beta.py").write_text("class SearchServiceLegacy:\n    pass\n", encoding="utf-8")
    fixture_path = tmp_path / "code_intel_cases.json"
    fixture_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "workspace-ordering",
                    "scenario_type": "workspace_symbol_ambiguity",
                    "language": "python",
                    "operation": "workspace_symbol",
                    "symbol": "SearchService",
                    "expected_substrings": ["SearchService"],
                    "expected_ordered_substrings": ["alpha.py", "beta.py"],
                    "unexpected_substrings": ["gamma.py"],
                }
            ]
        ),
        encoding="utf-8",
    )

    metrics = benchmark_code_intel(workspace, fixture_path)

    assert metrics["summary"]["case_count"] == 1
    assert metrics["cases"][0]["passed"] is True
    assert metrics["cases"][0]["assertion_counts"]["ordered"] == 2
    assert metrics["cases"][0]["assertion_counts"]["unexpected"] == 1


def test_ci_workflow_runs_code_intel_quality_benchmark() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "Run code_intel quality benchmark" in workflow
    assert "python benchmarks/code_intel_quality_benchmark.py" in workflow
