"""
Compare pipeline for cross-document analysis.

This flow is intentionally separate from chat routing/state.
"""

from __future__ import annotations

from typing import Dict, List
import json
import logging
import re

from services import get_document_service
from .generator import call_llm_api
from .retriever import retrieve
from .vectordb import get_document_chunks

logger = logging.getLogger(__name__)

MAX_COMPARE_CONTEXT_CHUNKS = 8

SECTION_HINTS = {
    "achievements": [
        "achievement",
        "achievements",
        "award",
        "awards",
        "accomplishment",
        "accomplishments",
        "honor",
        "honors",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "intern",
        "internship",
    ],
    "projects": [
        "project",
        "projects",
    ],
    "skills": [
        "skill",
        "skills",
        "technology",
        "technologies",
        "tech stack",
    ],
    "education": [
        "education",
        "academic",
        "university",
        "college",
        "school",
    ],
}

INTENT_STOPWORDS = {
    "compare",
    "comparison",
    "between",
    "across",
    "versus",
    "with",
    "about",
    "the",
    "this",
    "that",
    "these",
    "those",
    "document",
    "documents",
}


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _tokenize(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (value or "").lower()) if len(token) > 2}


def _intent_keywords(intent: str) -> List[str]:
    words = []
    seen = set()
    for token in re.findall(r"[a-z0-9]+", (intent or "").lower()):
        if len(token) < 3 or token in INTENT_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        words.append(token)
    return words


def extract_intent(query: str) -> str:
    """
    Extract the focused comparison intent from a user query.

    Example:
        "compare achievements" -> "achievements"
    """
    cleaned = _clean_text(query)
    if not cleaned:
        return ""

    lowered = cleaned.lower()
    lowered = re.sub(
        r"\b(compare|comparison|between|across|vs|versus|the|these|this|documents?|document|of)\b",
        " ",
        lowered,
    )
    lowered = re.sub(r"\s+", " ", lowered).strip(" ,.-")
    return lowered or cleaned


def _extract_json_blob(raw_text: str) -> str | None:
    if not raw_text:
        return None

    cleaned = raw_text.strip()
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            return fenced_match.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1].strip()

    return None


def _coerce_string_list(values: object) -> List[str]:
    if not isinstance(values, list):
        return []
    output: List[str] = []
    seen = set()
    for item in values:
        text = _clean_text(item)
        if not text:
            continue
        marker = text.lower()
        if marker in seen:
            continue
        seen.add(marker)
        output.append(text)
    return output


def _coerce_topic_items(values: object, default_topic: str) -> List[Dict[str, str]]:
    if not isinstance(values, list):
        return []

    output: List[Dict[str, str]] = []
    seen = set()

    for item in values:
        topic = default_topic
        details = ""

        if isinstance(item, dict):
            topic = _clean_text(item.get("topic") or default_topic)
            details = _clean_text(item.get("details") or item.get("summary") or item.get("description") or "")
        else:
            details = _clean_text(item)

        if not details:
            continue

        topic = topic or default_topic
        marker = f"{topic.lower()}::{details.lower()}"
        if marker in seen:
            continue
        seen.add(marker)

        output.append({
            "topic": topic[:120],
            "details": details,
        })

    return output


def _normalize_summary_payload(payload: object) -> Dict | None:
    if not isinstance(payload, dict):
        return None

    overview = _clean_text(payload.get("overview") or "")
    similarities = _coerce_topic_items(payload.get("similarities"), default_topic="Common Ground")
    differences = payload.get("differences") if isinstance(payload.get("differences"), dict) else {}

    doc_a = _coerce_topic_items(differences.get("docA"), default_topic="Document A")
    doc_b = _coerce_topic_items(differences.get("docB"), default_topic="Document B")

    if not overview:
        if similarities:
            overview = similarities[0]["details"][:280]
        else:
            overview = "Comparison generated from retrieved multi-chunk evidence."

    normalized = {
        "overview": overview,
        "similarities": similarities,
        "differences": {
            "docA": doc_a,
            "docB": doc_b,
        },
    }

    return normalized


def _parse_summary_json(raw_text: str) -> Dict | None:
    blob = _extract_json_blob(raw_text)
    if not blob:
        return None

    try:
        parsed = json.loads(blob)
    except Exception:
        return None

    return _normalize_summary_payload(parsed)


def _chunk_page(chunk: Dict) -> int:
    metadata = chunk.get("metadata", {}) or {}
    try:
        return int(chunk.get("page", metadata.get("page", metadata.get("page_number", 1))) or 1)
    except Exception:
        return 1


def _chunk_section_value(chunk: Dict) -> str:
    metadata = chunk.get("metadata", {}) or {}
    return _clean_text(metadata.get("section") or metadata.get("section_title") or "").lower()


def _normalize_chunk_payload(chunk: Dict) -> Dict:
    metadata = chunk.get("metadata", {}) or {}
    page = _chunk_page(chunk)
    return {
        **chunk,
        "text": chunk.get("text") or "",
        "metadata": metadata,
        "page": page,
        "doc_name": chunk.get("doc_name")
        or metadata.get("doc_name")
        or metadata.get("document_name")
        or metadata.get("file_name")
        or "Unknown",
        "score": float(chunk.get("score", chunk.get("rerank_score", chunk.get("relevance_score", 1.0)))),
    }


def _chunk_matches_section(chunk: Dict, section: str) -> bool:
    aliases = SECTION_HINTS.get(section, [section])

    section_value = _chunk_section_value(chunk)
    if any(alias in section_value for alias in aliases):
        return True

    first_line = _clean_text((chunk.get("text") or "").splitlines()[0] if (chunk.get("text") or "") else "").lower()
    if any(alias in first_line for alias in aliases):
        return True

    return False


def _keyword_overlap_score(text: str, intent: str) -> float:
    keywords = _intent_keywords(intent)
    if not keywords:
        return 0.0

    lowered = (text or "").lower()
    hits = sum(1 for kw in keywords if kw in lowered)
    return hits / max(1, len(keywords))


def _score_chunk_for_intent(chunk: Dict, intent: str) -> float:
    snippet = _clean_text((chunk.get("text") or "")[:1200])
    lexical = _keyword_overlap_score(snippet, intent)
    retrieval = float(chunk.get("score", chunk.get("rerank_score", chunk.get("relevance_score", 0.0))))
    return (0.75 * lexical) + (0.25 * retrieval)


def _split_candidate_lines(text: str) -> List[str]:
    if not text:
        return []

    parts: List[str] = []
    for line in str(text).splitlines():
        cleaned = _clean_text(line)
        if cleaned:
            parts.append(cleaned)

    if parts:
        return parts

    sentence_parts = re.split(r"(?<=[.!?])\s+", str(text))
    return [_clean_text(part) for part in sentence_parts if _clean_text(part)]


def extract_lines(text: str, intent: str, max_lines: int = 2) -> List[str]:
    lines = _split_candidate_lines(text)
    if not lines:
        return []

    keywords = _intent_keywords(intent)
    filtered = []

    for line in lines:
        lowered = line.lower()
        if keywords and any(keyword in lowered for keyword in keywords):
            filtered.append(line)

    candidates = filtered if filtered else lines[: max_lines + 1]

    compact = []
    for line in candidates:
        bounded = _clean_text(line)[:220]
        if len(bounded) < 18:
            continue
        compact.append(bounded)
        if len(compact) >= max_lines:
            break

    return compact


def extract_supporting_text(chunks: List[Dict], intent: str, max_items: int = 8) -> List[Dict]:
    """
    Extract section/block-level highlight candidates from retrieved chunks.

    Returned strings are verbatim chunk substrings so frontend can do direct
    block matching against page text.
    """
    if not chunks:
        return []

    highlights: List[Dict] = []
    seen = set()

    sorted_chunks = sorted(chunks, key=lambda c: _score_chunk_for_intent(c, intent), reverse=True)

    for chunk in sorted_chunks:
        page = _chunk_page(chunk)
        for line in extract_lines(chunk.get("text") or "", intent=intent, max_lines=2):
            marker = (str(page), line.lower())
            if marker in seen:
                continue
            seen.add(marker)
            highlights.append({
                "text": line,
                "page": page,
            })
            if len(highlights) >= max_items:
                return highlights

    if highlights:
        return highlights

    for chunk in sorted_chunks:
        page = _chunk_page(chunk)
        fallback = _clean_text((chunk.get("text") or "")[:220])
        if len(fallback) < 18:
            continue
        marker = (str(page), fallback.lower())
        if marker in seen:
            continue
        seen.add(marker)
        highlights.append({"text": fallback, "page": page})
        if len(highlights) >= max_items:
            break

    return highlights


def _is_general_comparison_query(query: str, intent: str) -> bool:
    lowered = f"{query} {intent}".lower()
    general_markers = [
        "compare both",
        "both documents",
        "overall",
        "general",
        "all aspects",
        "everything",
        "full comparison",
    ]
    if any(marker in lowered for marker in general_markers):
        return True

    cleaned_intent = _clean_text(intent).lower()
    return cleaned_intent in {"", "both", "both documents", "documents"}


def _filter_chunks_for_comparison(chunks: List[Dict], intent: str, query: str, max_chunks: int) -> List[Dict]:
    if not chunks:
        return []

    ranking_text = intent or query
    ranked = sorted(chunks, key=lambda chunk: _score_chunk_for_intent(chunk, ranking_text), reverse=True)

    keywords = _intent_keywords(intent)
    if not keywords and not _is_general_comparison_query(query, intent):
        keywords = _intent_keywords(query)

    matched: List[Dict] = []
    if keywords:
        matched = [chunk for chunk in ranked if _keyword_overlap_score(chunk.get("text") or "", ranking_text) > 0]

    selected: List[Dict] = []
    seen = set()
    for chunk in matched + ranked:
        marker = id(chunk)
        if marker in seen:
            continue
        seen.add(marker)
        selected.append(chunk)
        if len(selected) >= max_chunks:
            break

    return selected


def _chunk_to_context_block(chunk: Dict, index: int) -> str:
    text = _clean_text(chunk.get("text") or "")
    if not text:
        return ""

    section = _chunk_section_value(chunk)
    page = _chunk_page(chunk)
    heading = f"[Chunk {index} | Page {page}"
    if section:
        heading += f" | Section {section}"
    heading += "]"

    bounded_text = text[:1400]
    if len(text) > 1400:
        bounded_text += " ..."

    return f"{heading}\n{bounded_text}"


def _build_comparison_context(chunks: List[Dict], intent: str, query: str, max_chunks: int) -> tuple[List[Dict], str]:
    filtered_chunks = _filter_chunks_for_comparison(chunks, intent, query, max_chunks=max_chunks)

    context_blocks = []
    for idx, chunk in enumerate(filtered_chunks, start=1):
        block = _chunk_to_context_block(chunk, idx)
        if block:
            context_blocks.append(block)

    return filtered_chunks, "\n\n".join(context_blocks)


async def llm_compare(doc_a_chunks: List[Dict], doc_b_chunks: List[Dict], query: str, intent: str) -> Dict:
    """Generate final structured comparison summary from multi-chunk context."""
    filtered_chunks_a, doc_a_context = _build_comparison_context(
        doc_a_chunks,
        intent=intent,
        query=query,
        max_chunks=MAX_COMPARE_CONTEXT_CHUNKS,
    )
    filtered_chunks_b, doc_b_context = _build_comparison_context(
        doc_b_chunks,
        intent=intent,
        query=query,
        max_chunks=MAX_COMPARE_CONTEXT_CHUNKS,
    )

    comparison_input = {
        "docA": filtered_chunks_a,
        "docB": filtered_chunks_b,
    }

    filtered_chunks_a = comparison_input["docA"]
    filtered_chunks_b = comparison_input["docB"]

    if not doc_a_context:
        doc_a_context = "No relevant chunks retrieved for Document A."
    if not doc_b_context:
        doc_b_context = "No relevant chunks retrieved for Document B."

    print("[DOC A CHUNKS]:", len(filtered_chunks_a))
    print("[DOC B CHUNKS]:", len(filtered_chunks_b))
    print("[DOC A PREVIEW]:", doc_a_context[:500])
    print("[DOC B PREVIEW]:", doc_b_context[:500])

    focus_topic = _clean_text(intent) or _clean_text(query) or "general comparison"
    if _is_general_comparison_query(query, intent):
        scope_instruction = "The query is general. Cover ALL meaningful aspects across both documents."
    else:
        scope_instruction = f"The query is topic-specific. ONLY focus on: \"{focus_topic}\"."

    compare_prompt = f"""You are comparing two documents in depth.

Analyze BOTH documents thoroughly and produce a DETAILED comparison.

Cover ALL relevant aspects such as:
- purpose
- key concepts
- methods / approaches
- technologies
- results / outcomes
- strengths
- limitations

{scope_instruction}

Return STRICT JSON:
{{
  "overview": "high level comparison summary",
  "similarities": [
    {{
      "topic": "topic name",
      "details": "detailed explanation"
    }}
  ],
  "differences": {{
    "docA": [
      {{
        "topic": "topic name",
        "details": "detailed explanation"
      }}
    ],
    "docB": [
      {{
        "topic": "topic name",
        "details": "detailed explanation"
      }}
    ]
  }}
}}

Rules:
- JSON only. No markdown, no prose, no code fences.
- Build the comparison only from the provided contexts.
- Keep each topic specific and avoid duplicates.
- Make details evidence-grounded and concrete.

User query:
{query}

Detected intent:
{focus_topic}

Document A context:
{doc_a_context}

Document B context:
{doc_b_context}
""".strip()

    raw = await call_llm_api(
        compare_prompt,
        max_tokens=2000,
        temperature=0.1,
    )
    parsed = _parse_summary_json(raw)
    if parsed:
        return parsed

    raise ValueError("Could not parse structured compare summary JSON")


def _fallback_comparison_summary(doc_a_chunks: List[Dict], doc_b_chunks: List[Dict], query: str, intent: str) -> Dict:
    """Fallback summary when compare LLM call is unavailable."""
    focus = _clean_text(intent) or _clean_text(query) or "the requested topic"
    left = _clean_text((doc_a_chunks[0].get("text") if doc_a_chunks else ""))[:320]
    right = _clean_text((doc_b_chunks[0].get("text") if doc_b_chunks else ""))[:320]

    return {
        "overview": f"Both documents were compared for {focus}.",
        "similarities": [
            {
                "topic": "Shared focus",
                "details": f"Both documents discuss {focus}.",
            }
        ],
        "differences": {
            "docA": [
                {
                    "topic": "Document A emphasis",
                    "details": left or "No clear difference extracted for Document A.",
                }
            ],
            "docB": [
                {
                    "topic": "Document B emphasis",
                    "details": right or "No clear difference extracted for Document B.",
                }
            ],
        },
    }


def _resolve_doc_name(document_id: str) -> str:
    service = get_document_service()
    doc = service.get_document(document_id)
    if not doc:
        return document_id
    return str(doc.get("file_name") or document_id)


def _build_document_brief(chunks: List[Dict], intent: str, max_points: int = 3) -> str:
    if not chunks:
        return "No relevant information found in this document for the comparison query."

    ranked = sorted(chunks, key=lambda chunk: _score_chunk_for_intent(chunk, intent), reverse=True)

    points = []
    seen = set()
    for chunk in ranked[:MAX_COMPARE_CONTEXT_CHUNKS]:
        for line in extract_lines(chunk.get("text") or "", intent=intent, max_lines=1):
            marker = line.lower()
            if marker in seen:
                continue
            seen.add(marker)
            points.append(f"- {line}")
            if len(points) >= max_points:
                break
        if len(points) >= max_points:
            break

    if points:
        return "\n".join(points)

    fallback = _clean_text((ranked[0].get("text") or ""))[:320]
    return fallback or "Relevant chunks were retrieved but no concise preview was extracted."


def _detect_compare_section(query: str, intent: str) -> str | None:
    lowered = f"{query} {intent}".lower()
    for section, aliases in SECTION_HINTS.items():
        if any(alias in lowered for alias in aliases):
            return section
    return None


def _retrieve_section_chunks(document_id: str, section: str, intent: str, top_k: int) -> List[Dict]:
    source_chunks = get_document_chunks(
        [document_id],
        max_chunks_per_document=max(140, top_k * 40),
    )
    normalized = [_normalize_chunk_payload(chunk) for chunk in source_chunks]

    filtered = [chunk for chunk in normalized if _chunk_matches_section(chunk, section)]
    if not filtered:
        return []

    filtered.sort(key=lambda chunk: _score_chunk_for_intent(chunk, intent), reverse=True)
    return filtered[:top_k]


def _retrieve_for_document(document_id: str, intent: str, query: str, top_k: int) -> List[Dict]:
    section = _detect_compare_section(query, intent)
    if section:
        section_chunks = _retrieve_section_chunks(document_id, section, intent, top_k=top_k)
        if section_chunks:
            print(f"[RETRIEVAL MODE:{document_id}]", "section")
            print(f"[TARGET SECTION:{document_id}]", section)
            return section_chunks

    chunks = retrieve(
        intent or query,
        top_k=top_k,
        selected_document_ids=[document_id],
    )

    if not chunks and (intent or "").strip() and intent.strip().lower() != query.strip().lower():
        chunks = retrieve(
            query,
            top_k=top_k,
            selected_document_ids=[document_id],
        )

    normalized = [_normalize_chunk_payload(chunk) for chunk in chunks]
    print(f"[RETRIEVAL MODE:{document_id}]", "semantic")
    print(f"[TARGET SECTION:{document_id}]", section or "None")
    return normalized[:top_k]


async def run_compare_pipeline(doc_ids: List[str], query: str, top_k: int = 5) -> Dict:
    """
    Execute standalone compare flow.

    Steps:
    1. Extract intent
    2. Retrieve top-k chunks per document
    3. Build multi-chunk context and generate comparison summary
    4. Build short document previews
    5. Extract supporting highlight text
    """
    if not doc_ids or len(doc_ids) < 2:
        return {
            "success": False,
            "error": "At least two documents are required for comparison.",
        }

    doc_a_id, doc_b_id = doc_ids[0], doc_ids[1]
    intent = extract_intent(query)

    print("[COMPARE QUERY]:", query)
    print("[INTENT]:", intent)

    retrieval_top_k = max(int(top_k or 0), MAX_COMPARE_CONTEXT_CHUNKS)
    filtered_chunks_a = _retrieve_for_document(doc_a_id, intent, query, top_k=retrieval_top_k)
    filtered_chunks_b = _retrieve_for_document(doc_b_id, intent, query, top_k=retrieval_top_k)

    print("[DOC A CHUNKS]:", len(filtered_chunks_a))
    print("[DOC B CHUNKS]:", len(filtered_chunks_b))

    doc_a_text = _build_document_brief(filtered_chunks_a, intent)
    doc_b_text = _build_document_brief(filtered_chunks_b, intent)

    try:
        summary = await llm_compare(filtered_chunks_a, filtered_chunks_b, query, intent)
    except Exception as e:
        logger.warning(f"Compare summary generation failed, using fallback: {str(e)}")
        summary = _fallback_comparison_summary(filtered_chunks_a, filtered_chunks_b, query, intent)

    # Highlights are source-aligned from retrieved chunks only (not LLM answers).
    highlights_a = extract_supporting_text(filtered_chunks_a, intent)
    highlights_b = extract_supporting_text(filtered_chunks_b, intent)

    return {
        "success": True,
        "summary": summary,
        "intent": intent,
        "docA": {
            "doc_id": doc_a_id,
            "doc_name": _resolve_doc_name(doc_a_id),
            "text": doc_a_text,
            "highlights": highlights_a,
        },
        "docB": {
            "doc_id": doc_b_id,
            "doc_name": _resolve_doc_name(doc_b_id),
            "text": doc_b_text,
            "highlights": highlights_b,
        },
    }