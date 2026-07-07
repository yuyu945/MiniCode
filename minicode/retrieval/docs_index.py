from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from minicode.retrieval.docs_chunking import chunk_markdown_document
from minicode.retrieval.docs_discovery import discover_documents
from minicode.retrieval.docs_types import ChildChunk, DocumentRecord, ParentChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass(slots=True)
class DocsIndex:
    workspace: Path
    documents: list[DocumentRecord] = field(default_factory=list)
    parents: list[ParentChunk] = field(default_factory=list)
    children: list[ChildChunk] = field(default_factory=list)
    parent_by_id: dict[str, ParentChunk] = field(default_factory=dict)

    def build(self) -> DocsIndex:
        self.documents = discover_documents(self.workspace)
        self.parents = []
        self.children = []
        self.parent_by_id = {}

        for record in self.documents:
            text = (self.workspace / record.path).read_text(encoding="utf-8")
            parents, children = chunk_markdown_document(
                path=record.path,
                doc_id=record.doc_id,
                text=text,
            )
            for parent in parents:
                parent.last_modified_at = record.last_modified_at
                self.parents.append(parent)
                self.parent_by_id[parent.parent_id] = parent
            self.children.extend(children)

        return self

    def sparse_search(
        self,
        query: str,
        top_k: int = 10,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[tuple[ChildChunk, float]]:
        query_terms = _query_terms(query)
        if not query_terms:
            return []

        scored: list[tuple[ChildChunk, float]] = []
        for child in self.children:
            if allowed_doc_ids is not None and child.doc_id not in allowed_doc_ids:
                continue
            title_terms = _query_terms(" ".join(child.title_path))
            content_terms = _query_terms(child.content)
            matched_title_terms = query_terms.intersection(title_terms)
            matched_content_terms = query_terms.intersection(content_terms)
            if not matched_title_terms and not matched_content_terms:
                continue

            score = float(len(matched_content_terms) + (1.5 * len(matched_title_terms)))
            if matched_title_terms and matched_content_terms:
                score += 0.5
            scored.append((child, score))

        scored.sort(key=lambda item: (-item[1], item[0].ordinal, item[0].child_id))
        return scored[:top_k]

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[tuple[ChildChunk, float]]:
        query_terms = _query_terms(query)
        if not query_terms:
            return []

        scored: list[tuple[ChildChunk, float]] = []
        for child in self.children:
            if allowed_doc_ids is not None and child.doc_id not in allowed_doc_ids:
                continue
            overlap = query_terms.intersection(set(child.keywords))
            if not overlap:
                continue
            score = float(0.5 + (0.2 * len(overlap)))
            scored.append((child, score))

        scored.sort(key=lambda item: (-item[1], item[0].ordinal, item[0].child_id))
        return scored[:top_k]


def _query_terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}
