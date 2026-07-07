import shutil
import uuid
from pathlib import Path

from minicode.retrieval.docs_types import (
    ChildChunk,
    DocumentRecord,
    DocsRetrievalResult,
    ParentChunk,
)
from minicode.retrieval import (
    ChildChunk as ExportedChildChunk,
    DocumentRecord as ExportedDocumentRecord,
    DocsRetrievalResult as ExportedDocsRetrievalResult,
    ParentChunk as ExportedParentChunk,
)


def test_docs_types_support_parent_child_hierarchy() -> None:
    doc = DocumentRecord(
        doc_id="readme",
        path="README.md",
        doc_type="readme",
        title="MiniCode",
        tags=["root"],
        last_modified_at=123.0,
        content_hash="abc",
    )
    parent = ParentChunk(
        parent_id="readme::architecture",
        doc_id=doc.doc_id,
        path=doc.path,
        title_path=["MiniCode", "Architecture"],
        heading="Architecture",
        heading_level=2,
        content="Architecture section",
        token_count=120,
        tags=["architecture"],
        last_modified_at=123.0,
    )
    child = ChildChunk(
        child_id="readme::architecture::0",
        parent_id=parent.parent_id,
        doc_id=doc.doc_id,
        path=doc.path,
        title_path=parent.title_path,
        ordinal=0,
        content="Architecture section",
        token_count=120,
        start_offset=0,
        end_offset=21,
        keywords=["architecture"],
        embedding_ref=None,
    )
    result = DocsRetrievalResult(
        query="how is architecture described",
        matched_children=[child],
        expanded_parents=[parent],
        applied_filters={"doc_type": ["readme"]},
        ranking_signals={"readme::architecture::0": {"lexical": 0.9}},
        source="docs_pipeline",
        partition="project_docs",
    )

    assert result.expanded_parents[0].heading == "Architecture"
    assert result.matched_children[0].parent_id == parent.parent_id


def test_docs_types_are_reexported_from_retrieval_package() -> None:
    assert ExportedChildChunk is ChildChunk
    assert ExportedDocumentRecord is DocumentRecord
    assert ExportedDocsRetrievalResult is DocsRetrievalResult
    assert ExportedParentChunk is ParentChunk


def test_discover_documents_collects_readme_docs_and_agents() -> None:
    from minicode.retrieval.docs_discovery import discover_documents

    workspace = Path.cwd() / f"docs-discovery-{uuid.uuid4().hex}"
    workspace.mkdir()
    try:
        (workspace / "README.md").write_text("# Root\n", encoding="utf-8")
        (workspace / "docs").mkdir()
        (workspace / "docs" / "design.md").write_text("# Design\n", encoding="utf-8")
        (workspace / "nested").mkdir()
        (workspace / "nested" / "AGENTS.md").write_text("# Rules\n", encoding="utf-8")

        records = discover_documents(workspace)
        paths = {record.path for record in records}
        types = {record.doc_type for record in records}

        assert "README.md" in paths
        assert "docs/design.md" in paths
        assert "nested/AGENTS.md" in paths
        assert "readme" in types
        assert "agents" in types
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def test_docs_pipeline_returns_matching_parent_section_not_file_prefix(tmp_path: Path) -> None:
    from minicode.retrieval.docs_pipeline import DocsRetrievalPipeline

    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "README.md").write_text(
        "# MiniCode\n\n"
        "Intro text.\n\n"
        "## Setup\n\n"
        "Setup details.\n\n"
        "## Retrieval\n\n"
        "Parent child retrieval expands matching chunks to the full section.\n",
        encoding="utf-8",
    )

    pipeline = DocsRetrievalPipeline(workspace)
    pipeline.build_index()
    result = pipeline.retrieve("how does parent child retrieval work", max_results=3)

    assert result.expanded_parents
    assert result.expanded_parents[0].heading == "Retrieval"
    assert "Parent child retrieval" in result.expanded_parents[0].content


def test_docs_pipeline_doc_type_filter_is_applied_before_candidate_truncation(
    tmp_path: Path,
) -> None:
    from minicode.retrieval.docs_pipeline import DocsRetrievalPipeline

    workspace = tmp_path / "repo"
    workspace.mkdir()

    for index in range(10):
        (workspace / f"README-{index}.md").write_text(
            "# Noise\n\n"
            "## Retrieval\n\n"
            "Parent child retrieval work details.\n",
            encoding="utf-8",
        )

    (workspace / "docs").mkdir()
    (workspace / "docs" / "architecture.md").write_text(
        "# Architecture\n\n"
        "## Retrieval\n\n"
        "Parent child retrieval work details.\n",
        encoding="utf-8",
    )

    pipeline = DocsRetrievalPipeline(workspace)
    pipeline.build_index()
    result = pipeline.retrieve(
        "parent child retrieval work",
        max_results=3,
        filters={"doc_type": ["design"]},
    )

    assert result.expanded_parents
    assert [parent.path for parent in result.expanded_parents] == ["docs/architecture.md"]
