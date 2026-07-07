from __future__ import annotations

from pathlib import Path

from minicode.retrieval.docs_index import DocsIndex
from minicode.retrieval.docs_types import ChildChunk, DocsRetrievalResult


class DocsRetrievalPipeline:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.index = DocsIndex(self.workspace)

    def build_index(self) -> None:
        self.index.build()

    def retrieve(
        self,
        query: str,
        *,
        max_results: int = 10,
        filters: dict[str, list[str]] | None = None,
    ) -> DocsRetrievalResult:
        applied_filters = filters or {}
        allowed_doc_ids = self._allowed_doc_ids(applied_filters)
        sparse_scores = self.index.sparse_search(
            query,
            top_k=max_results * 3,
            allowed_doc_ids=allowed_doc_ids,
        )
        semantic_scores = self.index.semantic_search(
            query,
            top_k=max_results * 3,
            allowed_doc_ids=allowed_doc_ids,
        )
        child_scores = self._rrf_merge(sparse_scores, semantic_scores)
        filtered_scores = self._apply_filters(child_scores, applied_filters)
        limited_scores = filtered_scores[:max_results]

        matched_children = [child for child, _ in limited_scores]
        expanded_parents = []
        seen_parent_ids: set[str] = set()
        ranking_signals: dict[str, dict[str, float]] = {}
        lexical_by_child_id = {child.child_id: score for child, score in sparse_scores}
        semantic_by_child_id = {child.child_id: score for child, score in semantic_scores}

        for child, fused_score in limited_scores:
            ranking_signals[child.child_id] = {
                "lexical": lexical_by_child_id.get(child.child_id, 0.0),
                "semantic": semantic_by_child_id.get(child.child_id, 0.0),
                "rrf": fused_score,
            }
            if child.parent_id in seen_parent_ids:
                continue
            parent = self.index.parent_by_id.get(child.parent_id)
            if parent is None:
                continue
            seen_parent_ids.add(child.parent_id)
            expanded_parents.append(parent)

        return DocsRetrievalResult(
            query=query,
            matched_children=matched_children,
            expanded_parents=expanded_parents,
            applied_filters=applied_filters,
            ranking_signals=ranking_signals,
        )

    def _rrf_merge(
        self,
        sparse_scores: list[tuple[ChildChunk, float]],
        semantic_scores: list[tuple[ChildChunk, float]],
    ) -> list[tuple[ChildChunk, float]]:
        fused: dict[str, tuple[ChildChunk, float]] = {}

        for rank, (child, _) in enumerate(sparse_scores, start=1):
            fused[child.child_id] = (child, 1.0 / (60 + rank))

        for rank, (child, _) in enumerate(semantic_scores, start=1):
            score = 1.0 / (60 + rank)
            existing = fused.get(child.child_id)
            if existing is None:
                fused[child.child_id] = (child, score)
                continue
            fused[child.child_id] = (child, existing[1] + score)

        merged = list(fused.values())
        merged.sort(key=lambda item: (-item[1], item[0].ordinal, item[0].child_id))
        return merged

    def _apply_filters(
        self,
        child_scores: list[tuple[ChildChunk, float]],
        filters: dict[str, list[str]],
    ) -> list[tuple[ChildChunk, float]]:
        allowed_doc_types = set(filters.get("doc_type", []))
        if not allowed_doc_types:
            return child_scores

        doc_type_by_doc_id = {document.doc_id: document.doc_type for document in self.index.documents}
        filtered: list[tuple[ChildChunk, float]] = []
        for child, score in child_scores:
            if doc_type_by_doc_id.get(child.doc_id) in allowed_doc_types:
                filtered.append((child, score))
        return filtered

    def _allowed_doc_ids(self, filters: dict[str, list[str]]) -> set[str] | None:
        allowed_doc_types = set(filters.get("doc_type", []))
        if not allowed_doc_types:
            return None
        return {
            document.doc_id
            for document in self.index.documents
            if document.doc_type in allowed_doc_types
        }
