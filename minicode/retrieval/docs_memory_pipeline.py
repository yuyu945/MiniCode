from __future__ import annotations

from pathlib import Path


class DocsMemoryRetrievalPipeline:
    def __init__(self, workspace_path: str | None, memory_manager) -> None:
        self.workspace = Path(workspace_path) if workspace_path else None
        self.memory = memory_manager

    def retrieve(
        self,
        query: str,
        active_domains: list[str] | None = None,
        max_results: int = 10,
    ) -> list[dict]:
        results: list[dict] = []
        if self.workspace and self.workspace.exists():
            results.extend(self._retrieve_docs(query, active_domains=active_domains, max_results=max_results))
        if self.memory is not None:
            results.extend(self._retrieve_memory(query, active_domains=active_domains, max_results=max_results))
        results.sort(key=lambda item: item["relevance"], reverse=True)
        return results[:max_results]

    def _retrieve_docs(self, query: str, active_domains: list[str] | None, max_results: int) -> list[dict]:
        assert self.workspace is not None
        query_terms = _query_terms(query)
        results: list[dict] = []
        doc_paths = []
        doc_paths.extend(self.workspace.glob("README*"))
        docs_dir = self.workspace / "docs"
        if docs_dir.exists():
            doc_paths.extend(docs_dir.rglob("*.md"))
        doc_paths.extend(self.workspace.rglob("AGENTS.md"))
        for path in sorted({item for item in doc_paths if item.is_file()}):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            lowered = text.lower()
            matched_terms = [term for term in query_terms if term and term in lowered]
            if not matched_terms:
                continue
            relevance = 0.4 + 0.1 * len(matched_terms)
            results.append(
                {
                    "id": path.relative_to(self.workspace).as_posix(),
                    "content": text[:300],
                    "domain": active_domains or [],
                    "relevance": relevance,
                    "source": "docs_pipeline",
                    "partition": "project_docs",
                }
            )
            if len(results) >= max_results:
                break
        return results

    def _retrieve_memory(self, query: str, active_domains: list[str] | None, max_results: int) -> list[dict]:
        results: list[dict] = []
        entries = self.memory.search(query, limit=max_results, active_domains=active_domains)
        if not entries:
            expanded_terms = sorted(_query_terms(query))
            normalized_query = " ".join(expanded_terms)
            if normalized_query and normalized_query != query:
                entries = self.memory.search(normalized_query, limit=max_results, active_domains=active_domains)
            if not entries:
                seen_ids: set[str] = set()
                for term in expanded_terms:
                    for scope in list(getattr(self.memory, "memories", {}).keys()):
                        for entry in self.memory.search_by_tag(scope, term):
                            if entry.id not in seen_ids:
                                seen_ids.add(entry.id)
                                entries.append(entry)
                                if len(entries) >= max_results:
                                    break
                        if len(entries) >= max_results:
                            break
                    if len(entries) >= max_results:
                        break
        for entry in entries:
            partition = "recent_memory"
            freshness = getattr(entry, "freshness", "")
            if freshness == "stale" or getattr(entry, "conflicts_with", None):
                partition = "stale_or_conflict_memory"
            elif getattr(entry, "usage_count", 0) > 5:
                partition = "historical_memory"
            results.append(
                {
                    "id": entry.id,
                    "content": entry.content,
                    "domain": getattr(entry, "domains", []),
                    "relevance": float(getattr(entry, "usage_count", 0)) + 0.3,
                    "source": "memory_pipeline",
                    "partition": partition,
                }
            )
        return results


def _query_terms(query: str) -> set[str]:
    terms: set[str] = set()
    expansions = {
        "test": {"pytest", "testing"},
        "tests": {"test", "pytest", "testing"},
        "testing": {"test", "pytest"},
        "api": {"fastapi", "backend"},
        "apis": {"api", "fastapi", "backend"},
    }
    for raw in query.lower().split():
        token = raw.strip(" ,.:;!?()[]{}\"'")
        if not token:
            continue
        terms.add(token)
        if token.endswith("s") and len(token) > 4:
            terms.add(token[:-1])
        if token in expansions:
            terms.update(expansions[token])
    return terms
