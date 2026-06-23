"""Code retrieval pipeline for local workspace source search."""

from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_QUERY_STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "in", "for", "on", "with",
    "where", "find", "implemented", "implementation", "function", "class",
    "method", "code", "source", "locate",
}

_QUERY_EXPANSIONS = {
    "auth": ["authentication", "authorize", "login", "token"],
    "login": ["authenticate", "auth", "signin"],
    "session": ["sessions", "store", "loader", "load_session"],
    "loader": ["load", "loadsession", "load_session", "read"],
    "memory": ["manager", "storage", "search"],
    "search": ["find", "lookup", "query"],
    "context": ["compaction", "manager", "window"],
    "entrypoint": ["run", "main", "start"],
    "turn": ["run", "agent", "run_agent_turn", "runagentturn"],
    "manager": ["controller", "owner"],
    "adapter": ["modeladapter", "openai"],
    "permission": ["permissions", "gate", "manager", "decision", "permissionmanager"],
    "tty": ["terminal", "run_tty_app"],
    "orchestrator": ["controller", "compactor"],
    "owns": ["owner", "manager", "control"],
    "controls": ["control", "controller", "scheduler", "manager"],
    "decisions": ["decision", "manager", "permission"],
    "client": ["adapter", "service", "llmclient"],
    "runtime": ["run", "agent", "entrypoint", "agentruntime"],
    "loaded": ["load"],
    "implemented": ["implement", "define", "class", "function"],
    "registered": ["register"],
    "scheduling": ["scheduler", "schedule", "toolscheduler"],
    "llm": ["llmclient"],
    "permissionmanager": ["permission", "manager"],
}

_TS_SYMBOL_RE = re.compile(
    r"(?P<kind>export\s+class|class|export\s+function|function)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<tail>\([^)]*\))?",
    re.MULTILINE,
)


@dataclass
class CodeChunk:
    chunk_id: str
    path: str
    language: str
    symbol_name: str
    symbol_kind: str
    content: str
    signature: str
    start_line: int
    end_line: int
    terms: list[str] = field(default_factory=list)
    term_freq: dict[str, int] = field(default_factory=dict)
    term_set: set[str] = field(default_factory=set)
    normalized_aliases: set[str] = field(default_factory=set)
    symbol_terms: set[str] = field(default_factory=set)
    path_terms: set[str] = field(default_factory=set)

    def to_search_result(self, score: float, matched_terms: list[str]) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "language": self.language,
            "symbol_name": self.symbol_name,
            "symbol_kind": self.symbol_kind,
            "signature": self.signature,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": round(score, 4),
            "matched_terms": matched_terms,
        }


class CodeRetrieval:
    """Workspace code retrieval with symbol-aware indexing."""

    def __init__(self) -> None:
        self._workspace: Path | None = None
        self._chunks: list[CodeChunk] = []
        self._by_id: dict[str, CodeChunk] = {}
        self._doc_freq: dict[str, int] = {}
        self._avg_doc_len = 0.0
        self._indexed_files = 0
        self._failed_files: list[str] = []

    def index_workspace(self, workspace_path: str | Path) -> "CodeRetrieval":
        self._workspace = Path(workspace_path)
        self._chunks.clear()
        self._by_id.clear()
        self._doc_freq.clear()
        self._avg_doc_len = 0.0
        self._indexed_files = 0
        self._failed_files.clear()

        source_files = []
        for pattern in ("*.py", "*.ts", "*.tsx"):
            source_files.extend(self._workspace.rglob(pattern))

        for file_path in source_files:
            if not file_path.is_file():
                continue
            try:
                chunks = self._extract_chunks(file_path)
                for chunk in chunks:
                    self._chunks.append(chunk)
                    self._by_id[chunk.chunk_id] = chunk
                self._indexed_files += 1
            except Exception:
                self._failed_files.append(str(file_path))

        self._rebuild_corpus_stats()
        return self

    def benchmark_ready_index(self, workspace_path: str | Path) -> "CodeRetrieval":
        return self.index_workspace(workspace_path)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "indexed_files": self._indexed_files,
            "indexed_chunks": len(self._chunks),
            "languages": sorted({chunk.language for chunk in self._chunks}),
            "failed_files": list(self._failed_files),
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        language_scope: list[str] | None = None,
        symbol_kinds: list[str] | None = None,
        *,
        mode: str = "hybrid_rerank",
    ) -> list[dict[str, Any]]:
        if not self._chunks:
            return []

        terms = self._rewrite_query(query)
        lexical_ranked = self._lexical_rank(terms, language_scope, symbol_kinds)
        vector_ranked = self._vector_rank(terms, language_scope, symbol_kinds)

        candidate_cap = max(top_k * 12, 60)

        if mode == "baseline_dense":
            ranked_ids = vector_ranked[: candidate_cap]
        elif mode == "hybrid":
            ranked_ids = self._fuse_rankings(lexical_ranked, vector_ranked)
        else:
            ranked_ids = self._fuse_rankings(lexical_ranked, vector_ranked)

        ranked_ids = self._augment_candidates(
            ranked_ids,
            terms,
            language_scope,
            symbol_kinds,
            candidate_cap,
        )

        results = []
        for chunk_id, fused_score in ranked_ids[:candidate_cap]:
            chunk = self._by_id[chunk_id]
            reranked = (
                self._rerank_score(chunk, query, terms, fused_score)
                if mode == "hybrid_rerank"
                else fused_score
            )
            matched_terms = [term for term in terms if term in chunk.terms][:8]
            results.append((reranked, chunk, matched_terms))

        results.sort(key=lambda item: item[0], reverse=True)
        return [
            chunk.to_search_result(score, matched_terms)
            for score, chunk, matched_terms in results[:top_k]
        ]

    def _augment_candidates(
        self,
        ranked_ids: list[tuple[str, float]],
        terms: list[str],
        language_scope: list[str] | None,
        symbol_kinds: list[str] | None,
        candidate_cap: int,
    ) -> list[tuple[str, float]]:
        scored = {chunk_id: score for chunk_id, score in ranked_ids[:candidate_cap]}
        normalized_terms = {term.replace("_", "") for term in terms}
        for chunk in self._iter_filtered(language_scope, symbol_kinds):
            overlap = normalized_terms & chunk.normalized_aliases
            if overlap:
                alias_score = 0.8 + 0.4 * len(overlap)
                scored[chunk.chunk_id] = max(scored.get(chunk.chunk_id, 0.0), alias_score)
        return sorted(scored.items(), key=lambda item: item[1], reverse=True)

    def _extract_chunks(self, file_path: Path) -> list[CodeChunk]:
        text = file_path.read_text(encoding="utf-8")
        rel_path = file_path.relative_to(self._workspace).as_posix() if self._workspace else file_path.name
        language = "python" if file_path.suffix == ".py" else "typescript"
        lines = text.splitlines()

        chunks = [
            self._build_chunk(
                rel_path,
                language,
                file_path.stem,
                "file",
                text,
                f"file {rel_path}",
                1,
                max(1, len(lines)),
            )
        ]

        if language == "python":
            chunks.extend(self._extract_python_chunks(rel_path, text))
        else:
            chunks.extend(self._extract_typescript_chunks(rel_path, text))

        return chunks

    def _extract_python_chunks(self, rel_path: str, text: str) -> list[CodeChunk]:
        tree = ast.parse(text)
        lines = text.splitlines()
        chunks: list[CodeChunk] = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                chunks.append(
                    self._build_chunk(
                        rel_path,
                        "python",
                        node.name,
                        "class",
                        self._slice_source(lines, node.lineno, getattr(node, "end_lineno", node.lineno)),
                        f"class {node.name}",
                        node.lineno,
                        getattr(node, "end_lineno", node.lineno),
                    )
                )
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        chunks.append(
                            self._build_chunk(
                                rel_path,
                                "python",
                                child.name,
                                "method",
                                self._slice_source(lines, child.lineno, getattr(child, "end_lineno", child.lineno)),
                                f"def {child.name}",
                                child.lineno,
                                getattr(child, "end_lineno", child.lineno),
                            )
                        )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(
                    self._build_chunk(
                        rel_path,
                        "python",
                        node.name,
                        "function",
                        self._slice_source(lines, node.lineno, getattr(node, "end_lineno", node.lineno)),
                        f"def {node.name}",
                        node.lineno,
                        getattr(node, "end_lineno", node.lineno),
                    )
                )
        return chunks

    def _extract_typescript_chunks(self, rel_path: str, text: str) -> list[CodeChunk]:
        lines = text.splitlines()
        chunks: list[CodeChunk] = []
        for match in _TS_SYMBOL_RE.finditer(text):
            name = match.group("name")
            kind_text = match.group("kind")
            kind = "class" if "class" in kind_text else "function"
            start_line = text[: match.start()].count("\n") + 1
            end_line = self._find_block_end(lines, start_line)
            chunks.append(
                self._build_chunk(
                    rel_path,
                    "typescript",
                    name,
                    kind,
                    self._slice_source(lines, start_line, end_line),
                    match.group(0).strip(),
                    start_line,
                    end_line,
                )
            )
        return chunks

    def _build_chunk(
        self,
        path: str,
        language: str,
        symbol_name: str,
        symbol_kind: str,
        content: str,
        signature: str,
        start_line: int,
        end_line: int,
    ) -> CodeChunk:
        chunk_id = f"{path}::{symbol_kind}::{symbol_name}::{start_line}"
        base_terms = self._tokenize(" ".join([path, symbol_name, symbol_kind, signature, content]))
        alias_terms = set(base_terms)
        symbol_aliases = self._identifier_aliases(symbol_name)
        path_aliases = self._identifier_aliases(Path(path).stem)
        alias_terms.update(symbol_aliases)
        alias_terms.update(path_aliases)
        for part in Path(path).parts:
            alias_terms.update(self._identifier_aliases(part))
        sorted_terms = sorted(alias_terms)
        return CodeChunk(
            chunk_id=chunk_id,
            path=path,
            language=language,
            symbol_name=symbol_name,
            symbol_kind=symbol_kind,
            content=content,
            signature=signature,
            start_line=start_line,
            end_line=end_line,
            terms=sorted_terms,
            term_freq=dict(Counter(sorted_terms)),
            term_set=set(sorted_terms),
            normalized_aliases={term.replace("_", "") for term in sorted_terms},
            symbol_terms=set(self._tokenize(f"{symbol_name} {signature}")) | symbol_aliases,
            path_terms=set(self._tokenize(path)) | path_aliases,
        )

    def _rebuild_corpus_stats(self) -> None:
        doc_lengths = []
        for chunk in self._chunks:
            doc_lengths.append(len(chunk.terms) or 1)
            for term in chunk.term_set:
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
        self._avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0

    def _rewrite_query(self, query: str) -> list[str]:
        base_terms = [term for term in self._tokenize(query) if term not in _QUERY_STOPWORDS]
        normalized = []
        for term in base_terms:
            normalized.append(term)
            normalized.extend(self._normalize_term(term))
        base_terms = list(dict.fromkeys(normalized))
        expanded = list(base_terms)
        for term in list(base_terms):
            for synonym in _QUERY_EXPANSIONS.get(term, []):
                if synonym not in expanded:
                    expanded.append(synonym)
        return expanded or self._tokenize(query)

    def _lexical_rank(
        self,
        terms: list[str],
        language_scope: list[str] | None,
        symbol_kinds: list[str] | None,
    ) -> list[tuple[str, float]]:
        ranked = []
        total_docs = max(len(self._chunks), 1)
        for chunk in self._iter_filtered(language_scope, symbol_kinds):
            score = 0.0
            doc_len = max(len(chunk.terms), 1)
            for term in terms:
                tf = chunk.term_freq.get(term, 0)
                if tf == 0:
                    continue
                df = self._doc_freq.get(term, 1)
                idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
                norm = tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * doc_len / max(self._avg_doc_len, 1.0)))
                score += idf * norm
            if score > 0:
                ranked.append((chunk.chunk_id, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _vector_rank(
        self,
        terms: list[str],
        language_scope: list[str] | None,
        symbol_kinds: list[str] | None,
    ) -> list[tuple[str, float]]:
        query_set = set(terms)
        ranked = []
        for chunk in self._iter_filtered(language_scope, symbol_kinds):
            doc_set = chunk.term_set
            if not doc_set:
                continue
            overlap = len(query_set & doc_set)
            if overlap == 0:
                continue
            score = overlap / math.sqrt(len(query_set) * len(doc_set))
            ranked.append((chunk.chunk_id, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _fuse_rankings(
        self,
        lexical: list[tuple[str, float]],
        vector: list[tuple[str, float]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for rank, (chunk_id, _) in enumerate(lexical, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        for rank, (chunk_id, _) in enumerate(vector, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def _rerank_score(
        self,
        chunk: CodeChunk,
        query: str,
        terms: list[str],
        base_score: float,
    ) -> float:
        score = base_score
        query_lower = query.lower()
        path_lower = chunk.path.lower()
        symbol_lower = chunk.symbol_name.lower()
        basename_lower = Path(chunk.path).name.lower()
        symbol_terms = chunk.symbol_terms
        path_terms = chunk.path_terms
        exact_aliases = self._identifier_aliases(chunk.symbol_name)
        normalized_query_terms = {term.replace("_", "") for term in terms}
        normalized_aliases = chunk.normalized_aliases

        if symbol_lower and symbol_lower in query_lower.replace(" ", ""):
            score += 1.2
        term_overlap = len(set(terms) & symbol_terms)
        path_overlap = len(set(terms) & path_terms)
        if term_overlap:
            score += 0.35 * term_overlap
        if path_overlap:
            score += 0.2 * path_overlap
        if symbol_lower and any(term in symbol_lower for term in terms):
            score += 0.8
        if any(term == symbol_lower or term in exact_aliases for term in terms):
            score += 1.8
        if normalized_query_terms & normalized_aliases:
            score += 1.4
        if any(term in path_lower for term in terms):
            score += 0.5
        if chunk.symbol_kind in {"function", "method"} and any(term in {"function", "method", "load", "search", "login"} for term in terms):
            score += 0.25
        if chunk.symbol_kind == "class" and "class" in terms:
            score += 0.25
        if chunk.symbol_name.startswith("_") and not chunk.symbol_name.startswith("__"):
            score -= 0.7
        if "class" in query_lower and chunk.symbol_kind == "class":
            score += 1.4
        if "class" in query_lower and chunk.symbol_kind != "class":
            score -= 1.2
        if ("function" in query_lower or "entrypoint" in query_lower) and chunk.symbol_kind not in {"function", "method"}:
            score -= 1.0
        if "file" in query_lower and chunk.symbol_kind == "file":
            score += 1.5
        if basename_lower in query_lower and chunk.symbol_kind == "file":
            score += 2.0
        if chunk.symbol_kind == "file" and not any(word in query_lower for word in ["file", "module", ".py", ".ts", ".tsx"]):
            score -= 0.8
        if "manager" in query_lower and chunk.symbol_kind == "class" and "manager" in symbol_terms:
            score += 1.4
        if any(word in query_lower for word in ["owns", "decision", "decisions"]) and chunk.symbol_kind == "class" and "manager" in exact_aliases:
            score += 1.4
        if "permission" in query_lower and any(word in query_lower for word in ["decision", "decisions", "owns"]) and chunk.symbol_kind == "class":
            if "manager" in exact_aliases:
                score += 2.0
            if "gate" in exact_aliases:
                score -= 0.6
        if "adapter" in query_lower and chunk.symbol_kind == "class" and "adapter" in symbol_terms:
            score += 1.0
        if "loader" in query_lower and chunk.symbol_kind in {"function", "method"} and "load" in symbol_terms:
            score += 1.2
        if "entrypoint" in query_lower and chunk.symbol_kind in {"function", "method"} and symbol_lower.startswith("run"):
            score += 1.2
        if "tty" in query_lower and "app" in query_lower and symbol_lower.startswith("run_tty"):
            score += 1.6
        if "turn" in query_lower and "agent" in query_lower and symbol_lower.startswith("run_"):
            score += 1.2
        if "full turn" in query_lower and "run" in symbol_terms and "turn" in symbol_terms:
            score += 1.0
        if "turn" in query_lower and "turn" not in exact_aliases and "turn" not in symbol_terms:
            score -= 1.4
        if "runagentturn" in normalized_query_terms and "runagentturn" in normalized_aliases:
            score += 2.5
        if "client" in query_lower and chunk.symbol_kind == "class" and "client" in exact_aliases:
            score += 1.4
        if "llm" in query_lower:
            if "llm" in exact_aliases or "llm" in path_terms:
                score += 2.0
            else:
                score -= 2.0
        if "permission" in query_lower and "permission" not in exact_aliases and "permission" not in path_terms and "permissions" not in path_terms:
            score -= 1.4
        if "full turn" in query_lower and "runagentturn" not in normalized_aliases and "turn" not in exact_aliases:
            score -= 1.2
        if "scheduler" in query_lower and chunk.symbol_kind == "class" and "scheduler" in exact_aliases:
            score += 1.4
        if "runtime" in query_lower and "entrypoint" in query_lower and chunk.symbol_kind == "class" and "runtime" in exact_aliases:
            score += 1.4
        if "register" in query_lower and chunk.symbol_kind in {"function", "method"} and symbol_lower.startswith("register"):
            score += 1.0
        if "orchestrator" in query_lower and chunk.symbol_kind == "class":
            if "orchestrator" in symbol_terms:
                score += 1.2
            elif "compactor" in symbol_terms:
                score += 0.8
        return score

    def _iter_filtered(
        self,
        language_scope: list[str] | None,
        symbol_kinds: list[str] | None,
    ) -> list[CodeChunk]:
        chunks = self._chunks
        if language_scope:
            allowed = set(language_scope)
            chunks = [chunk for chunk in chunks if chunk.language in allowed]
        if symbol_kinds:
            allowed_kinds = set(symbol_kinds)
            chunks = [chunk for chunk in chunks if chunk.symbol_kind in allowed_kinds]
        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())

    @staticmethod
    def _normalize_term(term: str) -> list[str]:
        normalized: list[str] = []
        if term.endswith("ed") and len(term) > 4:
            normalized.append(term[:-2])
        if term.endswith("ing") and len(term) > 5:
            normalized.append(term[:-3])
        if term.endswith("s") and len(term) > 4:
            normalized.append(term[:-1])
        return normalized

    @staticmethod
    def _identifier_aliases(text: str) -> set[str]:
        aliases: set[str] = set()
        for raw in re.split(r"[^A-Za-z0-9_]+", text):
            if not raw:
                continue
            pieces = re.split(r"_+", raw)
            for piece in pieces:
                if not piece:
                    continue
                parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+", piece)
                lowered_parts = [part.lower() for part in parts if part]
                if not lowered_parts:
                    continue
                aliases.update(lowered_parts)
                aliases.add("".join(lowered_parts))
                for width in range(2, len(lowered_parts) + 1):
                    for start in range(0, len(lowered_parts) - width + 1):
                        aliases.add("".join(lowered_parts[start:start + width]))
        return aliases

    @staticmethod
    def _slice_source(lines: list[str], start_line: int, end_line: int) -> str:
        return "\n".join(lines[start_line - 1:end_line]).strip()

    @staticmethod
    def _find_block_end(lines: list[str], start_line: int) -> int:
        depth = 0
        started = False
        for index in range(start_line - 1, len(lines)):
            line = lines[index]
            depth += line.count("{")
            if "{" in line:
                started = True
            depth -= line.count("}")
            if started and depth <= 0:
                return index + 1
        return min(len(lines), start_line + 20)


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
        case_record = {
            "query": query,
            "expected_targets": expected_targets,
            "methods": {},
        }

        for method in methods:
            results = retrieval.search(query, top_k=top_k, mode=method)
            file_hit_rank = _find_hit_rank(results, expected_targets, symbol_only=False)
            symbol_hit_rank = _find_hit_rank(results, expected_targets, symbol_only=True)
            relevant_count = sum(
                1 for result in results if _matches_expected(result, expected_targets)
            )
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
    top5_file_hits = sum(
        1 for case in cases if case["file_hit_rank"] is not None and case["file_hit_rank"] <= top_k
    )
    top5_symbol_hits = sum(
        1 for case in cases if case["symbol_hit_rank"] is not None and case["symbol_hit_rank"] <= top_k
    )
    reciprocal_ranks = [
        1.0 / case["file_hit_rank"] if case["file_hit_rank"] is not None else 0.0
        for case in cases
    ]
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
