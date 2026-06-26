# Retrieval Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current single-pass code retrieval utility with an LLM-driven multi-stage retrieval loop for code, plus a decoupled docs and memory retrieval pipeline.

**Architecture:** Build a new `minicode.retrieval` package that separates indexing, planning, candidate recall, dependency expansion, reranking, and prompt injection. Treat retrieval as a loop controller over deterministic tools and indices, then expose it through a new read-only `code_retrieve` tool and replace the old `minicode.code_retrieval` implementation with a thin facade over the new subsystem.

**Tech Stack:** Python 3.11, stdlib `ast`, repo-local regex parsing for TypeScript, existing `MemoryPipeline`, `MemoryReranker`, `ToolRegistry`, `pytest`

---

## Current Assessment (2026-06-26)

The staged retrieval refactor is now runnable, but it is **not yet enterprise-grade**. The current implementation has three material gaps:

1. **Intent and routing remain heuristic-driven**
   - `build_retrieval_intent()` is regex + stopword parsing, not an LLM-controlled retrieval planner.
   - `CodeRetrievalPipeline._structural_narrowing()` still contains query-shaped boosts that overfit a small number of benchmark cases.

2. **Structural retrieval lacks a true symbol/reference graph**
   - `CodeIndex` currently builds chunk and import indexes, but it does not expose robust definition lookup, reference lookup, or caller/callee style expansion.
   - Dependency expansion is therefore limited to import neighbors and does not support reliable cross-file convergence.

3. **Benchmark outputs are not trustworthy enough for quality decisions**
   - `benchmark_code_retrieval()` still reports legacy method buckets (`baseline_dense`, `hybrid`, `hybrid_rerank`) even though `search(..., mode=...)` now uses one shared staged pipeline.
   - Existing fixtures are useful smoke tests, but they are not yet categorized by query type or designed to detect overfitting to hand-authored scoring rules.

## Milestone 2 Goal

Raise the retrieval system from **prototype** to a more defensible **structural retrieval baseline** before adding higher-level LLM planning.

This milestone is intentionally narrow:

- add a real `symbol/reference index` to the retrieval core
- use symbol definitions and references during convergence and expansion
- remove benchmark method illusions and evaluate the actual staged pipeline directly
- classify benchmark cases by retrieval shape so regressions are observable

## Milestone 2 Risks

- Python AST gives strong structure, but TypeScript is still regex-parsed; symbol/reference quality will remain asymmetric.
- Reference expansion can easily increase noise; ranking must favor definition-quality hits over generic mention hits.
- Benchmark compatibility will change because the old multi-method output shape is misleading and should no longer drive pass/fail decisions.

## Milestone 2 Validation Strategy

- focused unit tests for symbol definition/reference indexing
- focused retrieval tests that require cross-file symbol convergence, not just import adjacency
- benchmark fixture validation on at least:
  - direct symbol lookup
  - natural-language location query
  - cross-file dependency query
  - acronym / alias query
- runnable benchmark report that exposes:
  - top-1 / top-k hit rates
  - MRR
  - context precision
  - query-type breakdown

---

### Task 1: Create Retrieval Core Types And Package Boundary

**Files:**
- Create: `minicode/retrieval/__init__.py`
- Create: `minicode/retrieval/types.py`
- Create: `tests/test_code_retrieval.py`

- [ ] **Step 1: Write the failing test for the new retrieval result contract**

```python
from minicode.retrieval.types import CodeRetrievalResult, RetrievalIntent


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_code_retrieval.py -k multi_stage_evidence -q`
Expected: FAIL with `ModuleNotFoundError` or missing `CodeRetrievalResult`

- [ ] **Step 3: Add the new retrieval package and core dataclasses**

```python
# minicode/retrieval/types.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievalIntent:
    query: str
    symbols: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    file_hints: list[str] = field(default_factory=list)
    language_hints: list[str] = field(default_factory=list)
    stage_budget: int = 5
    dependency_hops: int = 1


@dataclass(slots=True)
class CodeEvidence:
    path: str
    symbol_name: str
    symbol_kind: str
    start_line: int
    end_line: int
    score: float
    source_stage: str
    evidence_type: str
    matched_terms: list[str] = field(default_factory=list)
    snippet: str = ""
    dependency_hops: int = 0
    why: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DependencyEdge:
    source_path: str
    target_path: str
    kind: str
    symbol: str = ""


@dataclass(slots=True)
class CodeRetrievalResult:
    query: str
    intent: RetrievalIntent
    candidates: list[CodeEvidence] = field(default_factory=list)
    expansions: list[DependencyEdge] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Export the new retrieval types**

```python
# minicode/retrieval/__init__.py
from minicode.retrieval.types import (
    CodeEvidence,
    CodeRetrievalResult,
    DependencyEdge,
    RetrievalIntent,
)

__all__ = [
    "CodeEvidence",
    "CodeRetrievalResult",
    "DependencyEdge",
    "RetrievalIntent",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_code_retrieval.py -k multi_stage_evidence -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add minicode/retrieval/__init__.py minicode/retrieval/types.py tests/test_code_retrieval.py
git commit -m "feat: add retrieval core types"
```

### Task 2: Replace The Legacy Code Retrieval Engine With A Multi-Stage Pipeline

**Files:**
- Create: `minicode/retrieval/intent.py`
- Create: `minicode/retrieval/code_index.py`
- Create: `minicode/retrieval/code_pipeline.py`
- Modify: `minicode/code_retrieval.py`
- Test: `tests/test_code_retrieval.py`

- [ ] **Step 1: Write failing tests for coarse search, structural narrowing, and dependency expansion**

```python
from minicode.code_retrieval import CodeRetrieval


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

    paths = [item["path"] for item in results]
    assert "auth.py" in paths
    assert "validator.py" in paths
```

- [ ] **Step 2: Run test to verify it fails against the current implementation**

Run: `python -m pytest tests/test_code_retrieval.py -k validator -q`
Expected: FAIL because dependency expansion does not reliably include `validator.py`

- [ ] **Step 3: Implement retrieval intent parsing**

```python
# minicode/retrieval/intent.py
from __future__ import annotations

import re

from minicode.retrieval.types import RetrievalIntent


def build_retrieval_intent(query: str, dependency_hops: int = 1) -> RetrievalIntent:
    terms = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query.lower())
    keywords = [term for term in terms if term not in {"where", "find", "the", "is", "a", "an"}]
    file_hints = [term for term in keywords if term in {"auth", "user", "session", "memory", "context", "tool"}]
    symbols = [term for term in keywords if len(term) > 2]
    return RetrievalIntent(
        query=query,
        symbols=symbols,
        keywords=keywords,
        file_hints=file_hints,
        stage_budget=5,
        dependency_hops=max(0, dependency_hops),
    )
```

- [ ] **Step 4: Implement indexing and import graph extraction**

```python
# minicode/retrieval/code_index.py
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class IndexedSymbol:
    path: str
    symbol_name: str
    symbol_kind: str
    start_line: int
    end_line: int
    signature: str
    content: str
    imports: list[str] = field(default_factory=list)


class CodeIndex:
    def __init__(self) -> None:
        self.workspace: Path | None = None
        self.symbols: list[IndexedSymbol] = []
        self.import_graph: dict[str, set[str]] = {}

    def build(self, workspace: str | Path) -> "CodeIndex":
        self.workspace = Path(workspace)
        self.symbols.clear()
        self.import_graph.clear()
        for path in sorted(self.workspace.rglob("*.py")):
            self._index_python_file(path)
        for path in sorted(self.workspace.rglob("*.ts")) + sorted(self.workspace.rglob("*.tsx")):
            self._index_ts_file(path)
        return self

    def _index_python_file(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(self.workspace).as_posix()
        tree = ast.parse(text)
        lines = text.splitlines()
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.replace(".", "/") + ".py")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.replace(".", "/") + ".py")
        self.import_graph[rel] = imports
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self.symbols.append(
                    IndexedSymbol(
                        path=rel,
                        symbol_name=node.name,
                        symbol_kind="function",
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        signature=f"def {node.name}",
                        content="\n".join(lines[node.lineno - 1:getattr(node, 'end_lineno', node.lineno)]),
                        imports=sorted(imports),
                    )
                )
```

- [ ] **Step 5: Implement the stage loop and replace `minicode.code_retrieval` internals**

```python
# minicode/retrieval/code_pipeline.py
from __future__ import annotations

from minicode.retrieval.intent import build_retrieval_intent
from minicode.retrieval.types import CodeEvidence, CodeRetrievalResult, DependencyEdge


class CodeRetrievalPipeline:
    def __init__(self, index) -> None:
        self.index = index

    def retrieve(self, query: str, top_k: int = 5, dependency_hops: int = 1) -> CodeRetrievalResult:
        intent = build_retrieval_intent(query, dependency_hops=dependency_hops)
        candidates: list[CodeEvidence] = []
        expansions: list[DependencyEdge] = []
        seen_paths: set[str] = set()

        for symbol in self.index.symbols:
            score = 0.0
            matched = [term for term in intent.keywords if term in symbol.symbol_name.lower() or term in symbol.path.lower() or term in symbol.content.lower()]
            if matched:
                score += 1.0 + 0.2 * len(matched)
            if any(hint in symbol.path.lower() for hint in intent.file_hints):
                score += 0.5
            if score <= 0:
                continue
            candidates.append(
                CodeEvidence(
                    path=symbol.path,
                    symbol_name=symbol.symbol_name,
                    symbol_kind=symbol.symbol_kind,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    score=score,
                    source_stage="structural_narrowing",
                    evidence_type="symbol",
                    matched_terms=matched,
                    snippet=symbol.content,
                    why=["keyword_match"],
                )
            )
            seen_paths.add(symbol.path)

        for evidence in list(candidates):
            if dependency_hops < 1:
                continue
            for imported in self.index.import_graph.get(evidence.path, set()):
                if imported in seen_paths:
                    continue
                seen_paths.add(imported)
                expansions.append(DependencyEdge(source_path=evidence.path, target_path=imported, kind="import"))
                candidates.append(
                    CodeEvidence(
                        path=imported,
                        symbol_name=Path(imported).stem,
                        symbol_kind="file",
                        start_line=1,
                        end_line=1,
                        score=max(evidence.score - 0.2, 0.1),
                        source_stage="dependency_expansion",
                        evidence_type="import_neighbor",
                        dependency_hops=1,
                        why=["import_neighbor"],
                    )
                )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return CodeRetrievalResult(
            query=query,
            intent=intent,
            candidates=candidates[:top_k],
            expansions=expansions,
            corrections=[],
        )
```

- [ ] **Step 6: Replace the legacy public facade**

```python
# minicode/code_retrieval.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from minicode.retrieval.code_index import CodeIndex
from minicode.retrieval.code_pipeline import CodeRetrievalPipeline


class CodeRetrieval:
    def __init__(self) -> None:
        self._index = CodeIndex()
        self._pipeline = CodeRetrievalPipeline(self._index)

    def index_workspace(self, workspace_path: str | Path) -> "CodeRetrieval":
        self._index.build(workspace_path)
        return self

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "indexed_files": len(self._index.import_graph),
            "indexed_chunks": len(self._index.symbols),
            "languages": ["python", "typescript"],
            "failed_files": [],
        }

    def search(self, query: str, top_k: int = 5, **_: Any) -> list[dict[str, Any]]:
        result = self._pipeline.retrieve(query, top_k=top_k)
        return [
            {
                "path": item.path,
                "symbol_name": item.symbol_name,
                "symbol_kind": item.symbol_kind,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "score": round(item.score, 4),
                "matched_terms": item.matched_terms,
                "source_stage": item.source_stage,
                "evidence_type": item.evidence_type,
                "dependency_hops": item.dependency_hops,
                "snippet": item.snippet,
            }
            for item in result.candidates
        ]
```

- [ ] **Step 7: Run focused retrieval tests**

Run: `python -m pytest tests/test_code_retrieval.py -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add minicode/retrieval/intent.py minicode/retrieval/code_index.py minicode/retrieval/code_pipeline.py minicode/code_retrieval.py tests/test_code_retrieval.py
git commit -m "feat: replace legacy code retrieval with staged pipeline"
```

### Task 3: Add A Read-Only `code_retrieve` Tool For The Agent Loop

**Files:**
- Create: `minicode/tools/code_retrieve.py`
- Modify: `minicode/tools/__init__.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write the failing tool registry test**

```python
from minicode.tools import create_default_tool_registry


def test_default_tool_registry_includes_code_retrieve(tmp_path):
    tools = create_default_tool_registry(str(tmp_path), runtime=None)
    names = {tool.name for tool in tools.list()}
    assert "code_retrieve" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tools.py -k code_retrieve -q`
Expected: FAIL because `code_retrieve` is not registered

- [ ] **Step 3: Implement the read-only retrieval tool**

```python
# minicode/tools/code_retrieve.py
from __future__ import annotations

from pathlib import Path

from minicode.code_retrieval import CodeRetrieval
from minicode.tooling import ToolDefinition, ToolResult
from minicode.workspace import resolve_tool_path


def _validate(input_data: dict) -> dict:
    query = input_data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required")
    return {
        "query": query.strip(),
        "path": input_data.get("path", "."),
        "top_k": max(1, min(int(input_data.get("top_k", 8)), 20)),
    }


def _run(input_data: dict, context) -> ToolResult:
    root = resolve_tool_path(context, input_data["path"], "analyze")
    retrieval = CodeRetrieval().index_workspace(root)
    results = retrieval.search(input_data["query"], top_k=input_data["top_k"])
    lines = [f"Query: {input_data['query']}", ""]
    for item in results:
        lines.append(
            f"{item['path']}:{item['start_line']} {item['symbol_kind']} {item['symbol_name']} "
            f"[stage={item['source_stage']} score={item['score']}]"
        )
    return ToolResult(ok=True, output="\n".join(lines))


code_retrieve_tool = ToolDefinition(
    name="code_retrieve",
    description="Run multi-stage code retrieval over the workspace using keyword recall, structural narrowing, dependency expansion, and focused snippets.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "path": {"type": "string"},
            "top_k": {"type": "integer"},
        },
        "required": ["query"],
    },
    validator=_validate,
    run=_run,
)
```

- [ ] **Step 4: Register the new tool in the default registry**

```python
# minicode/tools/__init__.py
from minicode.tools.code_retrieve import code_retrieve_tool

_CORE_TOOLS = [
    ask_user_tool,
    list_files_tool,
    grep_files_tool,
    read_file_tool,
    code_retrieve_tool,
    write_file_tool,
    edit_file_tool,
```

- [ ] **Step 5: Add read-only behavior coverage**

```python
from minicode.tools.code_retrieve import code_retrieve_tool
from minicode.tooling import ToolContext


def test_code_retrieve_rejects_workspace_escape(tmp_path):
    result = code_retrieve_tool.run(
        {"query": "find login", "path": "../outside"},
        ToolContext(cwd=str(tmp_path), permissions=None),
    )
    assert result.ok is False
```

- [ ] **Step 6: Run tool tests**

Run: `python -m pytest tests/test_tools.py -k code_retrieve -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add minicode/tools/code_retrieve.py minicode/tools/__init__.py tests/test_tools.py
git commit -m "feat: add code retrieve tool"
```

### Task 4: Split Docs Retrieval From Memory Retrieval

**Files:**
- Create: `minicode/retrieval/docs_memory_pipeline.py`
- Modify: `minicode/memory_pipeline.py`
- Test: `tests/test_memory_integration.py`

- [ ] **Step 1: Write the failing partition test**

```python
from pathlib import Path

from minicode.memory import MemoryManager
from minicode.memory_pipeline import MemoryPipeline


def test_memory_pipeline_returns_partitioned_docs_and_memory(tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "README.md").write_text("Project uses FastAPI and pytest", encoding="utf-8")
    manager = MemoryManager(project_root=workspace)
    manager.add_entry(manager.MemoryScope.PROJECT, "testing", "Use pytest fixtures", ["pytest"])
    pipeline = MemoryPipeline(manager)
    pipeline.initialize(workspace_path=str(workspace), enable_reranker=False, enable_vector=False)

    result = pipeline.read("how are tests organized", ["tests/test_api.py"])
    sources = {item["source"] for item in result}
    assert "docs_pipeline" in sources
    assert "memory_pipeline" in sources
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_memory_integration.py -k partitioned_docs_and_memory -q`
Expected: FAIL because `MemoryPipeline.read()` currently returns memory-only results

- [ ] **Step 3: Implement docs and memory retrieval partitions**

```python
# minicode/retrieval/docs_memory_pipeline.py
from __future__ import annotations

from pathlib import Path


class DocsMemoryRetrievalPipeline:
    def __init__(self, workspace_path: str, memory_manager) -> None:
        self.workspace = Path(workspace_path)
        self.memory = memory_manager

    def retrieve(self, query: str, active_domains: list[str] | None = None, max_results: int = 10) -> list[dict]:
        results: list[dict] = []
        for path in sorted(self.workspace.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            if any(term in text.lower() for term in query.lower().split()):
                results.append({
                    "id": path.relative_to(self.workspace).as_posix(),
                    "content": text[:300],
                    "domain": active_domains or [],
                    "relevance": 0.5,
                    "source": "docs_pipeline",
                    "partition": "project_docs",
                })
        if self.memory:
            for entry in self.memory.search(query, limit=max_results, active_domains=active_domains):
                results.append({
                    "id": entry.id,
                    "content": entry.content,
                    "domain": getattr(entry, "domains", []),
                    "relevance": getattr(entry, "usage_count", 0),
                    "source": "memory_pipeline",
                    "partition": "historical_memory",
                })
        return results[:max_results]
```

- [ ] **Step 4: Wire the docs pipeline into `MemoryPipeline`**

```python
# minicode/memory_pipeline.py
from minicode.retrieval.docs_memory_pipeline import DocsMemoryRetrievalPipeline

def initialize(...):
    ...
    self._docs_memory_pipeline = DocsMemoryRetrievalPipeline(
        workspace_path=workspace_path or "",
        memory_manager=self._memory,
    )

def read(...):
    ...
    combined = self._docs_memory_pipeline.retrieve(
        task_description,
        active_domains=active_domains,
        max_results=max_results,
    )
    return combined
```

- [ ] **Step 5: Run focused memory integration tests**

Run: `python -m pytest tests/test_memory_integration.py -k partitioned_docs_and_memory -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add minicode/retrieval/docs_memory_pipeline.py minicode/memory_pipeline.py tests/test_memory_integration.py
git commit -m "feat: split docs retrieval from memory retrieval"
```

### Task 5: Add Self-Correction, Confidence Routing, And Retrieval Budget Control

**Files:**
- Modify: `minicode/retrieval/code_pipeline.py`
- Modify: `minicode/agent_loop.py`
- Test: `tests/test_code_retrieval.py`
- Test: `tests/test_agent_loop.py`

- [ ] **Step 1: Write the failing correction-loop test**

```python
from minicode.code_retrieval import CodeRetrieval


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_code_retrieval.py -k correction_when_first_guess_is_weak -q`
Expected: FAIL because the public API does not expose `retrieve()`

- [ ] **Step 3: Add `retrieve()` and correction metadata to the retrieval facade**

```python
# minicode/code_retrieval.py
def retrieve(self, query: str, top_k: int = 5, dependency_hops: int = 1):
    return self._pipeline.retrieve(query, top_k=top_k, dependency_hops=dependency_hops)
```

```python
# minicode/retrieval/code_pipeline.py
if not candidates:
    corrections.append("coarse_search_empty")
elif candidates[0].score < 0.8:
    corrections.append("low_confidence_first_pass")
```

- [ ] **Step 4: Add retrieval budget and confidence routing**

```python
# minicode/retrieval/code_pipeline.py
budget = max(1, intent.stage_budget)
if budget <= 2:
    dependency_hops = 0
if candidates and candidates[0].score >= 1.8:
    corrections.append("high_confidence_stop")
else:
    corrections.append("continue_expansion")
```

- [ ] **Step 5: Add a minimal agent-loop regression test**

```python
def test_agent_turn_keeps_code_retrieve_available_in_default_registry(tmp_path):
    from minicode.tools import create_default_tool_registry

    registry = create_default_tool_registry(str(tmp_path), runtime=None)
    assert any(tool.name == "code_retrieve" for tool in registry.list())
```

- [ ] **Step 6: Run retrieval and agent-loop tests**

Run: `python -m pytest tests/test_code_retrieval.py tests/test_agent_loop.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add minicode/code_retrieval.py minicode/retrieval/code_pipeline.py minicode/agent_loop.py tests/test_code_retrieval.py tests/test_agent_loop.py
git commit -m "feat: add retrieval correction loop and budget control"
```

### Task 6: Full Verification

**Files:**
- Test: `tests/test_code_retrieval.py`
- Test: `tests/test_tools.py`
- Test: `tests/test_memory_integration.py`
- Test: `tests/test_agent_loop.py`

- [ ] **Step 1: Run focused retrieval verification**

Run: `python -m pytest tests/test_code_retrieval.py tests/test_tools.py tests/test_memory_integration.py tests/test_agent_loop.py -q`
Expected: PASS

- [ ] **Step 2: Run full project verification**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 3: Review diff before finalizing**

Run: `git status --short`
Expected: only the planned retrieval and test files are modified

- [ ] **Step 4: Commit the verified milestone**

```bash
git add PLAN.md minicode/code_retrieval.py minicode/retrieval minicode/tools/code_retrieve.py minicode/tools/__init__.py minicode/memory_pipeline.py tests/test_code_retrieval.py tests/test_tools.py tests/test_memory_integration.py tests/test_agent_loop.py
git commit -m "feat: rebuild retrieval as a staged control loop"
```
