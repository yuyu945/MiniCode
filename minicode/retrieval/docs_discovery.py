from __future__ import annotations

import hashlib
from pathlib import Path

from minicode.retrieval.docs_types import DocumentRecord


def discover_documents(workspace: str | Path) -> list[DocumentRecord]:
    root = Path(workspace)
    candidates: list[Path] = []
    candidates.extend(root.glob("README*"))

    docs_dir = root / "docs"
    if docs_dir.exists():
        candidates.extend(docs_dir.rglob("*.md"))

    candidates.extend(root.rglob("AGENTS.md"))

    records: list[DocumentRecord] = []
    seen: set[Path] = set()
    for path in sorted(candidates):
        if not path.is_file() or path in seen:
            continue

        seen.add(path)
        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(root).as_posix()

        records.append(
            DocumentRecord(
                doc_id=relative_path,
                path=relative_path,
                doc_type=_classify_doc_type(path, relative_path),
                title=_extract_title(text) or path.stem,
                tags=[],
                last_modified_at=path.stat().st_mtime,
                content_hash=hashlib.sha1(
                    text.encode("utf-8"),
                    usedforsecurity=False,
                ).hexdigest(),
            )
        )

    return records


def _classify_doc_type(path: Path, relative_path: str) -> str:
    lower_name = path.name.lower()
    lower_path = relative_path.lower()

    if lower_name.startswith("readme"):
        return "readme"
    if lower_name == "agents.md":
        return "agents"
    if "adr" in lower_path:
        return "adr"
    if "design" in lower_path or "architecture" in lower_path:
        return "design"
    return "general"


def _extract_title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""
