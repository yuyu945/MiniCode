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
