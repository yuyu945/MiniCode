from minicode.retrieval.docs_chunking import chunk_markdown_document


def test_chunk_markdown_document_preserves_heading_paths() -> None:
    text = (
        "# MiniCode\n\n"
        "Intro.\n\n"
        "## Architecture\n\n"
        "Architecture paragraph one.\n\n"
        "### Retrieval\n\n"
        "Retrieval paragraph.\n"
    )

    parents, children = chunk_markdown_document(
        path="README.md",
        doc_id="README.md",
        text=text,
        target_tokens=80,
        max_parent_tokens=160,
    )

    assert any(parent.heading == "Architecture" for parent in parents)
    retrieval_child = next(child for child in children if "Retrieval paragraph" in child.content)
    assert retrieval_child.title_path == ["MiniCode", "Architecture", "Retrieval"]


def test_chunk_markdown_document_splits_long_sections_into_multiple_children() -> None:
    text = "# Guide\n\n## Usage\n\n" + ("Paragraph.\n\n" * 50)
    parents, children = chunk_markdown_document(
        path="docs/guide.md",
        doc_id="docs/guide.md",
        text=text,
        target_tokens=40,
        max_parent_tokens=1000,
    )

    usage_parent = next(parent for parent in parents if parent.heading == "Usage")
    usage_children = [child for child in children if child.parent_id == usage_parent.parent_id]
    assert len(usage_children) > 1


def test_chunk_markdown_document_preserves_indented_code_block_content() -> None:
    text = (
        "# Guide\n\n"
        "## Example\n\n"
        "    def hello():\n"
        "        return 'world'\n"
    )

    parents, children = chunk_markdown_document(
        path="docs/guide.md",
        doc_id="docs/guide.md",
        text=text,
        target_tokens=80,
        max_parent_tokens=160,
    )

    example_parent = next(parent for parent in parents if parent.heading == "Example")
    example_child = next(child for child in children if child.parent_id == example_parent.parent_id)

    assert example_parent.content == "    def hello():\n        return 'world'"
    assert example_child.content == "    def hello():\n        return 'world'"


def test_chunk_markdown_document_splits_single_oversized_paragraph_when_parent_too_large() -> None:
    long_line = "word " * 300
    text = f"# Guide\n\n## Usage\n\n{long_line}"

    parents, children = chunk_markdown_document(
        path="docs/guide.md",
        doc_id="docs/guide.md",
        text=text,
        target_tokens=60,
        max_parent_tokens=120,
    )

    usage_parent = next(parent for parent in parents if parent.heading == "Usage")
    usage_children = [child for child in children if child.parent_id == usage_parent.parent_id]

    assert usage_parent.token_count > 120
    assert len(usage_children) > 1
