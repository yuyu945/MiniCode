from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    path: str
    doc_type: str
    title: str
    tags: list[str] = field(default_factory=list)
    last_modified_at: float = 0.0
    content_hash: str = ""


@dataclass(slots=True)
class ParentChunk:
    parent_id: str
    doc_id: str
    path: str
    title_path: list[str]
    heading: str
    heading_level: int
    content: str
    token_count: int
    tags: list[str] = field(default_factory=list)
    last_modified_at: float = 0.0


@dataclass(slots=True)
class ChildChunk:
    child_id: str
    parent_id: str
    doc_id: str
    path: str
    title_path: list[str]
    ordinal: int
    content: str
    token_count: int
    start_offset: int
    end_offset: int
    keywords: list[str] = field(default_factory=list)
    embedding_ref: str | None = None


@dataclass(slots=True)
class DocsRetrievalResult:
    query: str
    matched_children: list[ChildChunk] = field(default_factory=list)
    expanded_parents: list[ParentChunk] = field(default_factory=list)
    applied_filters: dict[str, list[str]] = field(default_factory=dict)
    ranking_signals: dict[str, dict[str, float]] = field(default_factory=dict)
    source: str = "docs_pipeline"
    partition: str = "project_docs"
