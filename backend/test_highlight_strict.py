from rag.pipeline import build_sources_from_citations, extract_highlights_from_sources


def _mk_words(text: str):
    words = []
    cursor = 0
    for idx, token in enumerate(text.split()):
        start = text.find(token, cursor)
        end = start + len(token)
        words.append(
            {
                "text": token,
                "char_start": start,
                "char_end": end,
                "bbox": [10.0 + idx * 5.0, 20.0, 14.0 + idx * 5.0, 24.0],
            }
        )
        cursor = end
    return words


def _mk_chunks():
    chunk_1_text = "ACHIEVEMENTS: Map an App line."
    chunk_2_text = "ACHIEVEMENTS: Technovista Hackathon winner."

    return [
        {
            "chunk_id": "chunk-1",
            "doc_name": "resume.pdf",
            "page": 1,
            "text": chunk_1_text,
            "score": 0.99,
            "words": _mk_words(chunk_1_text),
            "metadata": {"chunk_id": "chunk-1"},
        },
        {
            "chunk_id": "chunk-2",
            "doc_name": "resume.pdf",
            "page": 1,
            "text": chunk_2_text,
            "score": 0.95,
            "words": _mk_words(chunk_2_text),
            "metadata": {"chunk_id": "chunk-2"},
        },
    ]


def test_build_sources_from_citations_rejects_wrong_chunk_span():
    chunks = _mk_chunks()
    citations = [
        {"chunk_id": "chunk-1", "span": "Map an App"},
        # This span exists only in chunk-2, so chunk-1 citation must be rejected.
        {"chunk_id": "chunk-1", "span": "Technovista Hackathon"},
    ]

    sources = build_sources_from_citations(citations, chunks, max_sources=5)

    assert len(sources) == 1
    assert sources[0]["chunk_id"] == "chunk-1"
    assert sources[0]["answer_span"] == "Map an App"


def test_extract_highlights_from_sources_never_cross_matches_other_chunk():
    chunks = _mk_chunks()
    sources = [
        {
            "chunk_id": "chunk-1",
            # Exists only in chunk-2, so this must be skipped.
            "answer_span": "Technovista Hackathon",
            "text": "Technovista Hackathon",
            "doc": "resume.pdf",
            "file": "resume.pdf",
            "page": 1,
            "score": 0.95,
        },
        {
            "chunk_id": "chunk-1",
            "answer_span": "Map an App",
            "text": "Map an App",
            "doc": "resume.pdf",
            "file": "resume.pdf",
            "page": 1,
            "score": 0.99,
        },
    ]

    highlights = extract_highlights_from_sources(sources, chunks)

    assert len(highlights) == 1
    assert highlights[0]["text"] == "Map an App"
    assert highlights[0]["page"] == 1
    assert highlights[0]["doc_name"] == "resume.pdf"
    assert highlights[0]["boxes"]