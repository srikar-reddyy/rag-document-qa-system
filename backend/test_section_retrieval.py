from rag import retriever
from rag.pipeline import _build_embedding_input_text, build_section_mode_highlights


def test_detect_section_query_keywords():
    assert retriever.detect_section_query("about Conclusion") == "conclusion"
    assert retriever.detect_section_query("give literature review") == "literature review"
    assert retriever.detect_section_query("show results section") == "results"
    assert retriever.detect_section_query("what does this document say") is None


def test_retrieve_prioritizes_section_chunks_without_embedding_calls(monkeypatch):
    section_chunks = [
        {
            "text": "Conclusion\nThis section summarizes the findings.",
            "metadata": {
                "chunk_id": "chunk-conclusion-1",
                "doc_name": "paper.pdf",
                "page": 9,
                "section_title": "Conclusion",
                "section_title_normalized": "conclusion",
                "chunk_index": 3,
                "words_json": "[]",
            },
            "relevance_score": 1.0,
        },
        {
            "text": "Results\nThe model improves recall.",
            "metadata": {
                "chunk_id": "chunk-results-1",
                "doc_name": "paper.pdf",
                "page": 7,
                "section_title": "Results",
                "section_title_normalized": "results",
                "chunk_index": 1,
                "words_json": "[]",
            },
            "relevance_score": 1.0,
        },
    ]

    monkeypatch.setattr(
        retriever,
        "get_document_chunks",
        lambda document_ids, max_chunks_per_document=120: section_chunks,
    )

    def _fail_embed(_):
        raise AssertionError("embed_text should not be called when section chunks are found")

    def _fail_query_documents(*_, **__):
        raise AssertionError("query_documents should not be called when section chunks are found")

    monkeypatch.setattr(retriever, "embed_text", _fail_embed)
    monkeypatch.setattr(retriever, "query_documents", _fail_query_documents)

    output = retriever.retrieve(
        "about Conclusion",
        top_k=3,
        selected_document_ids=["doc-1"],
    )

    assert len(output) == 1
    assert output[0]["chunk_id"] == "chunk-conclusion-1"
    assert output[0]["metadata"]["section_title"] == "Conclusion"


def test_retrieve_falls_back_to_embeddings_when_no_section_match(monkeypatch):
    intro_only_chunks = [
        {
            "text": "Introduction\nThis section introduces the problem.",
            "metadata": {
                "chunk_id": "chunk-intro-1",
                "doc_name": "paper.pdf",
                "page": 1,
                "section_title": "Introduction",
                "section_title_normalized": "introduction",
                "chunk_index": 0,
                "words_json": "[]",
            },
            "relevance_score": 1.0,
        }
    ]

    semantic_results = {
        "documents": ["Results section shows a significant gain."],
        "metadatas": [
            {
                "chunk_id": "semantic-results-1",
                "doc_name": "paper.pdf",
                "page": 6,
                "section_title": "Results",
                "section_title_normalized": "results",
                "chunk_index": 2,
                "words_json": "[]",
            }
        ],
        "distances": [0.2],
    }

    monkeypatch.setattr(
        retriever,
        "get_document_chunks",
        lambda document_ids, max_chunks_per_document=120: intro_only_chunks,
    )
    monkeypatch.setattr(retriever, "embed_text", lambda _text: [1.0, 0.0])

    def _fake_query_documents(query_embedding, top_k=5, document_ids=None):
        return semantic_results

    monkeypatch.setattr(retriever, "query_documents", _fake_query_documents)

    output = retriever.retrieve(
        "show results section",
        top_k=2,
        selected_document_ids=["doc-1"],
    )

    assert len(output) == 1
    assert output[0]["chunk_id"] == "semantic-results-1"
    assert output[0]["metadata"]["section_title"] == "Results"


def test_retrieve_returns_full_contiguous_section_chunks(monkeypatch):
    ordered_chunks = [
        {
            "text": "Introduction\nThis introduces the topic.",
            "metadata": {
                "chunk_id": "chunk-intro-1",
                "doc_name": "paper.pdf",
                "page": 1,
                "section_title": "Introduction",
                "section_title_normalized": "introduction",
                "chunk_index": 0,
                "words_json": "[]",
            },
            "relevance_score": 1.0,
        },
        {
            "text": "This is the second paragraph of the introduction.",
            "metadata": {
                "chunk_id": "chunk-intro-2",
                "doc_name": "paper.pdf",
                "page": 1,
                "section_title": "Unknown",
                "section_title_normalized": "unknown",
                "chunk_index": 1,
                "words_json": "[]",
            },
            "relevance_score": 1.0,
        },
        {
            "text": "This is the third paragraph of the introduction.",
            "metadata": {
                "chunk_id": "chunk-intro-3",
                "doc_name": "paper.pdf",
                "page": 2,
                "section_title": "Unknown",
                "section_title_normalized": "unknown",
                "chunk_index": 2,
                "words_json": "[]",
            },
            "relevance_score": 1.0,
        },
        {
            "text": "Methodology\nThis section explains the approach.",
            "metadata": {
                "chunk_id": "chunk-method-1",
                "doc_name": "paper.pdf",
                "page": 2,
                "section_title": "Methodology",
                "section_title_normalized": "methodology",
                "chunk_index": 3,
                "words_json": "[]",
            },
            "relevance_score": 1.0,
        },
    ]

    monkeypatch.setattr(
        retriever,
        "get_document_chunks",
        lambda document_ids, max_chunks_per_document=120: ordered_chunks,
    )

    def _fail_embed(_):
        raise AssertionError("embed_text should not be called when contiguous section chunks are found")

    def _fail_query_documents(*_, **__):
        raise AssertionError("query_documents should not be called when contiguous section chunks are found")

    monkeypatch.setattr(retriever, "embed_text", _fail_embed)
    monkeypatch.setattr(retriever, "query_documents", _fail_query_documents)

    output = retriever.retrieve(
        "about introduction",
        top_k=1,
        selected_document_ids=["doc-1"],
    )

    assert len(output) == 3
    assert [item["chunk_id"] for item in output] == ["chunk-intro-1", "chunk-intro-2", "chunk-intro-3"]
    assert all(item.get("mode") == "section" for item in output)


def test_build_section_mode_highlights_groups_chunks_by_page():
    chunk_one_text = "Introduction heading text"
    chunk_two_text = "Introduction body paragraph"

    chunks = [
        {
            "chunk_id": "chunk-intro-1",
            "doc_name": "paper.pdf",
            "page": 1,
            "text": chunk_one_text,
            "mode": "section",
            "metadata": {
                "chunk_id": "chunk-intro-1",
                "doc_name": "paper.pdf",
                "page": 1,
                "section_title": "Introduction",
            },
            "words": [
                {"text": "Introduction", "char_start": 0, "char_end": 12, "bbox": [10, 10, 60, 20]},
                {"text": "heading", "char_start": 13, "char_end": 20, "bbox": [65, 10, 105, 20]},
            ],
        },
        {
            "chunk_id": "chunk-intro-2",
            "doc_name": "paper.pdf",
            "page": 1,
            "text": chunk_two_text,
            "mode": "section",
            "metadata": {
                "chunk_id": "chunk-intro-2",
                "doc_name": "paper.pdf",
                "page": 1,
                "section_title": "Unknown",
            },
            "words": [
                {"text": "body", "char_start": 13, "char_end": 17, "bbox": [10, 30, 38, 40]},
                {"text": "paragraph", "char_start": 18, "char_end": 27, "bbox": [42, 30, 95, 40]},
            ],
        },
    ]

    highlights = build_section_mode_highlights("introduction", chunks)

    assert len(highlights) == 1
    assert highlights[0]["mode"] == "section"
    assert highlights[0]["section"] == "introduction"
    assert highlights[0]["doc_name"] == "paper.pdf"
    assert highlights[0]["page"] == 1
    assert "Introduction heading text" in highlights[0]["text"]
    assert "Introduction body paragraph" in highlights[0]["text"]
    assert set(highlights[0]["chunk_ids"]) == {"chunk-intro-1", "chunk-intro-2"}
    assert len(highlights[0]["boxes"]) == 4


def test_build_embedding_input_text_prefixes_section_title():
    text = "This is the final explanation."
    metadata = {"section_title": "Conclusion"}

    embedding_input = _build_embedding_input_text(text, metadata)

    assert embedding_input == "Conclusion\nThis is the final explanation."


def test_build_embedding_input_text_skips_unknown_section_title():
    text = "Just plain content"

    assert _build_embedding_input_text(text, {"section_title": "Unknown"}) == text
    assert _build_embedding_input_text(text, {}) == text