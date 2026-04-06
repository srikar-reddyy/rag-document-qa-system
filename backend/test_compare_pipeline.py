import asyncio

from rag.compare_pipeline import extract_intent, extract_supporting_text, run_compare_pipeline


def test_extract_intent_strips_compare_words():
    assert extract_intent("compare achievements") == "achievements"
    assert extract_intent("Compare backend experience between documents") == "backend experience"
    assert extract_intent("  ") == ""


def test_extract_supporting_text_returns_chunk_blocks():
    chunks = [
        {
            "text": "Built backend APIs with FastAPI and PostgreSQL for high-volume products.",
            "score": 0.9,
            "page": 2,
        },
        {
            "text": "Worked on frontend polish and visual redesign for dashboards.",
            "score": 0.4,
            "page": 3,
        },
    ]

    highlights = extract_supporting_text(chunks, "backend APIs and services", max_items=2)

    assert len(highlights) >= 1
    assert "FastAPI" in highlights[0]["text"]
    assert highlights[0]["page"] == 2


def test_run_compare_pipeline_returns_structured_payload(monkeypatch):
    import rag.compare_pipeline as compare_pipeline

    def _fake_retrieve(query, top_k=5, selected_document_ids=None):
        doc_id = selected_document_ids[0]
        return [
            {
                "text": f"{doc_id} highlights backend API achievements and system design work.",
                "score": 0.8,
            }
        ]

    async def _fake_call_llm_api(prompt, max_tokens=2000, temperature=0.1):
        return (
            '{"overview":"Both documents cover backend engineering capabilities.",'
            '"similarities":[{"topic":"Backend APIs","details":"Both discuss API design and delivery."}],'
            '"differences":{"docA":[{"topic":"Career stage","details":"Doc A emphasizes internships."}],'
            '"docB":[{"topic":"Scale","details":"Doc B emphasizes production delivery scale."}]}}'
        )

    class _DummyDocService:
        def get_document(self, document_id):
            return {"document_id": document_id, "file_name": f"{document_id}.pdf"}

    monkeypatch.setattr(compare_pipeline, "retrieve", _fake_retrieve)
    monkeypatch.setattr(compare_pipeline, "call_llm_api", _fake_call_llm_api)
    monkeypatch.setattr(compare_pipeline, "get_document_chunks", lambda document_ids, max_chunks_per_document=40: [])
    monkeypatch.setattr(compare_pipeline, "get_document_service", lambda: _DummyDocService())

    result = asyncio.run(
        run_compare_pipeline(["doc-a", "doc-b", "doc-c"], "compare backend experience", top_k=5)
    )

    assert result["success"] is True
    assert result["docA"]["doc_id"] == "doc-a"
    assert result["docB"]["doc_id"] == "doc-b"
    assert result["docA"]["doc_name"] == "doc-a.pdf"
    assert result["docB"]["doc_name"] == "doc-b.pdf"
    assert isinstance(result["summary"], dict)
    assert result["summary"]["overview"]
    assert result["summary"]["similarities"][0]["topic"] == "Backend APIs"
    assert result["summary"]["differences"]["docA"][0]["details"]
    assert result["docA"]["highlights"]
    assert result["docB"]["highlights"]
    assert "text" in result["docA"]["highlights"][0]
    assert "page" in result["docA"]["highlights"][0]