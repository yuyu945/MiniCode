from __future__ import annotations

from pathlib import Path
from typing import Any

from minicode.retrieval.docs_pipeline import DocsRetrievalPipeline


class DocsMemoryRetrievalPipeline:
    def __init__(self, workspace_path: str | None, memory_manager: Any) -> None:
        self.workspace = Path(workspace_path) if workspace_path else None
        self.memory = memory_manager
        self.docs = None
        if self.workspace and self.workspace.exists():
            self.docs = DocsRetrievalPipeline(self.workspace)
            self.docs.build_index()

    def retrieve(
        self,
        query: str,
        active_domains: list[str] | None = None,
        max_results: int = 10,
    ) -> list[dict]:
        if self.docs is None:
            return []

        docs_result = self.docs.retrieve(query, max_results=max_results)
        results: list[dict] = []
        for rank, parent in enumerate(docs_result.expanded_parents, start=1):
            doc_bonus = 1.0 / (60 + rank)
            matched_child = next(
                (child for child in docs_result.matched_children if child.parent_id == parent.parent_id),
                None,
            )
            if matched_child is not None:
                doc_bonus = docs_result.ranking_signals.get(matched_child.child_id, {}).get("rrf", doc_bonus)
            relevance = 1.3 + doc_bonus
            results.append(
                {
                    "id": parent.parent_id,
                    "content": parent.content,
                    "path": parent.path,
                    "domain": active_domains or [],
                    "relevance": relevance,
                    "source": docs_result.source,
                    "partition": docs_result.partition,
                }
            )
        return results
