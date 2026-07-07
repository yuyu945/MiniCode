from minicode.retrieval.context_assembler import assemble_partitioned_context


def test_assemble_partitioned_context_keeps_docs_and_memory_separate() -> None:
    docs = [
        {
            "id": "README.md::retrieval",
            "content": "Docs section about retrieval",
            "path": "README.md",
            "source": "docs_pipeline",
            "partition": "project_docs",
        }
    ]
    memory = [
        {
            "id": "memory-1",
            "content": "Historical memory about pytest",
            "source": "memory_pipeline",
            "partition": "historical_memory",
        }
    ]

    context = assemble_partitioned_context(docs_results=docs, memory_results=memory)
    docs_start = context.index("## Docs Context")
    memory_start = context.index("## Memory Context")
    docs_section = context[docs_start:memory_start]
    memory_section = context[memory_start:]

    assert "## Docs Context" in context
    assert "## Memory Context" in context
    assert docs_start < memory_start
    assert "Docs section about retrieval" in docs_section
    assert "Historical memory about pytest" not in docs_section
    assert "Historical memory about pytest" in memory_section
    assert "Docs section about retrieval" not in memory_section


def test_assemble_partitioned_context_indents_multiline_content_within_bullets() -> None:
    context = assemble_partitioned_context(
        docs_results=[
            {
                "id": "README.md::retrieval",
                "content": "Docs first line\nDocs second line",
                "path": "README.md",
                "source": "docs_pipeline",
                "partition": "project_docs",
            }
        ],
        memory_results=[
            {
                "id": "memory-1",
                "content": "Memory first line\nMemory second line",
                "source": "memory_pipeline",
                "partition": "historical_memory",
            }
        ],
    )

    assert "- [README.md] Docs first line\n  Docs second line" in context
    assert "- [historical_memory] Memory first line\n  Memory second line" in context
