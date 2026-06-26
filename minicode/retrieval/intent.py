from __future__ import annotations

import re

from minicode.retrieval.types import RetrievalIntent

_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "where",
    "find",
    "implemented",
    "implementation",
    "defined",
    "define",
    "function",
    "class",
    "method",
    "code",
    "source",
    "locate",
}

_FILE_HINTS = {"auth", "user", "account", "session", "memory", "context", "tool", "permission", "runtime"}


def build_retrieval_intent(query: str, dependency_hops: int = 1, stage_budget: int = 5) -> RetrievalIntent:
    terms = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query.lower())
    keywords = [term for term in terms if term not in _STOPWORDS]
    file_hints = [term for term in keywords if term in _FILE_HINTS]
    symbols = [term for term in keywords if len(term) > 2]
    language_hints: list[str] = []
    if any(term in keywords for term in {"tsx", "typescript", "react"}):
        language_hints.append("typescript")
    if any(term in keywords for term in {"python", "pytest", "fastapi"}):
        language_hints.append("python")
    return RetrievalIntent(
        query=query,
        symbols=list(dict.fromkeys(symbols)),
        keywords=list(dict.fromkeys(keywords)),
        file_hints=list(dict.fromkeys(file_hints)),
        language_hints=language_hints,
        stage_budget=max(1, stage_budget),
        dependency_hops=max(0, dependency_hops),
    )
