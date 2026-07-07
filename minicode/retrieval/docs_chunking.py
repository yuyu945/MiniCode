from __future__ import annotations

import re
from collections.abc import Iterable

from minicode.retrieval.docs_types import ChildChunk, ParentChunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_STOPWORDS = {
    "about",
    "after",
    "before",
    "between",
    "into",
    "from",
    "with",
    "this",
    "that",
    "have",
    "will",
    "your",
    "guide",
    "paragraph",
}


def chunk_markdown_document(
    *,
    path: str,
    doc_id: str,
    text: str,
    target_tokens: int = 700,
    max_parent_tokens: int = 1100,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    sections = _split_markdown_sections(text)
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for section_index, section in enumerate(sections):
        content = section["content"]
        if not content.strip():
            continue

        parent_id = f"{doc_id}::section::{section_index}"
        token_count = _estimate_tokens(content)
        parent = ParentChunk(
            parent_id=parent_id,
            doc_id=doc_id,
            path=path,
            title_path=section["title_path"],
            heading=section["heading"],
            heading_level=section["level"],
            content=content,
            token_count=token_count,
            tags=[],
            last_modified_at=0.0,
        )
        parents.append(parent)

        child_segments = _split_into_children(content, target_tokens)
        should_split = token_count > max_parent_tokens or (
            token_count > target_tokens and len(child_segments) > 1
        )
        if not should_split:
            child_segments = [(0, len(content), content)]

        for ordinal, (start_offset, end_offset, child_content) in enumerate(child_segments):
            children.append(
                ChildChunk(
                    child_id=f"{parent_id}::child::{ordinal}",
                    parent_id=parent_id,
                    doc_id=doc_id,
                    path=path,
                    title_path=section["title_path"],
                    ordinal=ordinal,
                    content=child_content,
                    token_count=_estimate_tokens(child_content),
                    start_offset=start_offset,
                    end_offset=end_offset,
                    keywords=_extract_keywords(_iter_keywords_source(section["title_path"], child_content)),
                    embedding_ref=None,
                )
            )
    return parents, children


def _split_markdown_sections(text: str) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    title_stack: list[tuple[int, str]] = []
    current_heading = "Document"
    current_level = 0
    current_title_path = ["Document"]
    current_body: list[str] = []

    def flush() -> None:
        raw_content = "\n".join(current_body)
        if not raw_content.strip():
            return
        content = raw_content.strip("\n")
        sections.append(
            {
                "heading": current_heading,
                "level": current_level,
                "title_path": list(current_title_path),
                "content": content,
            }
        )

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            while title_stack and title_stack[-1][0] >= level:
                title_stack.pop()
            title_stack.append((level, heading))
            current_heading = heading
            current_level = level
            current_title_path = [title for _, title in title_stack]
            current_body = []
            continue
        current_body.append(line)

    flush()
    return sections


def _split_into_children(content: str, target_tokens: int) -> list[tuple[int, int, str]]:
    paragraphs = _split_paragraphs(content)
    if not paragraphs:
        return [(0, len(content), content)]

    children: list[tuple[int, int, str]] = []
    current_start = paragraphs[0][0]
    current_end = paragraphs[0][1]
    current_tokens = 0

    for start, end, paragraph in paragraphs:
        paragraph_tokens = _estimate_tokens(paragraph)
        if paragraph_tokens > target_tokens:
            if current_tokens > 0:
                children.append((current_start, current_end, content[current_start:current_end].strip("\n")))
                current_tokens = 0
            children.extend(_split_oversized_span(content, start, end, target_tokens))
            current_start = end
            current_end = end
            continue
        if current_tokens > 0 and current_tokens + paragraph_tokens > target_tokens:
            child_content = content[current_start:current_end].strip("\n")
            children.append((current_start, current_end, child_content))
            current_start = start
            current_end = end
            current_tokens = paragraph_tokens
            continue
        if current_tokens == 0:
            current_start = start
        current_end = end
        current_tokens += paragraph_tokens

    if current_tokens > 0:
        child_content = content[current_start:current_end].strip("\n")
        children.append((current_start, current_end, child_content))
    return children


def _estimate_tokens(text: str) -> int:
    cjk_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_chars = len(text) - cjk_chars
    return max(1, int(ascii_chars / 4.0 + cjk_chars / 1.5))


def _extract_keywords(text: str | Iterable[str]) -> list[str]:
    if not isinstance(text, str):
        text = " ".join(text)
    words = [match.group(0).lower() for match in _WORD_RE.finditer(text)]
    ranked = sorted({word for word in words if len(word) > 3 and word not in _STOPWORDS})
    return ranked[:20]


def _iter_keywords_source(title_path: list[str], content: str) -> Iterable[str]:
    for title in title_path:
        yield title
    yield content


def _split_paragraphs(content: str) -> list[tuple[int, int, str]]:
    paragraphs: list[tuple[int, int, str]] = []
    start = 0
    for match in _PARAGRAPH_SPLIT_RE.finditer(content):
        end = match.start()
        raw_paragraph = content[start:end]
        if raw_paragraph.strip():
            paragraph = raw_paragraph.strip("\n")
            paragraphs.append((start, end, paragraph))
        start = match.end()
    raw_paragraph = content[start:]
    if raw_paragraph.strip():
        paragraph = raw_paragraph.strip("\n")
        paragraphs.append((start, len(content), paragraph))
    return paragraphs


def _split_oversized_span(content: str, start: int, end: int, target_tokens: int) -> list[tuple[int, int, str]]:
    span = content[start:end]
    if _estimate_tokens(span) <= target_tokens:
        return [(start, end, span.strip("\n"))]

    approx_chars = max(1, target_tokens * 4)
    chunks: list[tuple[int, int, str]] = []
    local_start = 0
    span_length = len(span)

    while local_start < span_length:
        remaining = span[local_start:]
        if _estimate_tokens(remaining) <= target_tokens:
            chunk_start = start + local_start
            chunk_end = end
            chunks.append((chunk_start, chunk_end, content[chunk_start:chunk_end].strip("\n")))
            break

        split_at = min(span_length, local_start + approx_chars)
        split_at = _find_split_offset(span, local_start, split_at)
        if split_at <= local_start:
            split_at = min(span_length, local_start + approx_chars)
        chunk_start = start + local_start
        chunk_end = start + split_at
        chunks.append((chunk_start, chunk_end, content[chunk_start:chunk_end].strip("\n")))
        local_start = split_at

    return chunks


def _find_split_offset(span: str, local_start: int, preferred_end: int) -> int:
    for index in range(preferred_end, local_start, -1):
        if span[index - 1].isspace():
            return index
    return preferred_end
