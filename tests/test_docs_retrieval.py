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
