"""
RAG Pipeline
Orchestrates the complete RAG workflow
"""

from typing import Dict, List
import logging
import re

from .loader import load_document
from .chunker import chunk_documents, Document
from .embedder import embed_texts
from .vectordb import add_documents, get_document_metadata, get_document_chunks, get_chunks_by_page
from .retriever import retrieve, extract_sources, detect_page_query
from .generator import generate_answer, call_llm_api
from .utils import clean_candidate_lines, score_sentence_relevance, split_sentences

logger = logging.getLogger(__name__)


def _to_int(value, default: int | None = None):
    try:
        return int(value)
    except Exception:
        return default


def _parse_bbox(value):
    """Parse bbox from list/tuple or comma-separated string into [x, y, w, h]."""
    if value is None:
        return None

    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except Exception:
            return None

    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) >= 4:
            try:
                return [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
            except Exception:
                return None

    return None


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _normalize_for_dedupe(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_query_about_experience(query: str) -> bool:
    q = (query or "").lower()
    return any(token in q for token in ["experience", "work", "intern", "job"])


def _section_boost(query: str, metadata: Dict) -> float:
    """Section-aware boosting (no hard filtering)."""
    section = str((metadata or {}).get("section") or (metadata or {}).get("section_title") or "").lower()
    if _is_query_about_experience(query) and "experience" in section:
        return 0.2
    return 0.0


def _chunk_to_source(chunk: Dict, score: float | None = None) -> Dict:
    metadata = chunk.get("metadata", {}) or {}
    text = (chunk.get("text") or "").strip()
    sentences = split_sentences(text)
    evidence_text = sentences[0] if sentences else (text[:260] + "..." if len(text) > 260 else text)

    doc_name = chunk.get("doc_name") or metadata.get("doc_name") or metadata.get("document_name") or metadata.get("file_name") or "Unknown"
    page = _to_int(chunk.get("page", metadata.get("page", metadata.get("page_number"))), 1) or 1
    bbox = _parse_bbox(chunk.get("bbox", metadata.get("bbox")))

    return {
        "doc": doc_name,
        "file": doc_name,
        "page": page,
        "text": evidence_text,
        "score": float(score if score is not None else chunk.get("score", chunk.get("rerank_score", chunk.get("relevance_score", 0.0)))),
        "query_similarity": float(chunk.get("rerank_score", chunk.get("score", 0.0))),
        "answer_similarity": 0.0,
        "bbox": bbox,
    }


def _fallback_sources_from_chunks(chunks: List[Dict], limit: int = 2) -> List[Dict]:
    if not chunks:
        return []
    ranked = sorted(
        chunks,
        key=lambda c: c.get("rerank_score", c.get("score", c.get("relevance_score", 0.0))),
        reverse=True,
    )
    return [_chunk_to_source(chunk) for chunk in ranked[:max(1, limit)]]


async def _extract_supporting_sentences(query: str, answer: str, chunk_text: str) -> List[str]:
    """
    LLM extraction of supporting sentences for source attribution.
    """
    if not chunk_text or not chunk_text.strip():
        return []

    prompt = f"""
Query: {query}
Answer: {answer}

From the text below, extract sentences that are likely relevant to the query and help justify the answer.

Rules:
- Return only likely relevant sentences
- No extra text
- No headings
- No unrelated lines
- One sentence per line

Text:
{chunk_text}
""".strip()

    try:
        raw = await call_llm_api(prompt)
        candidate_lines = [line.strip(" -•\t") for line in raw.splitlines() if line and line.strip()]
        cleaned = clean_candidate_lines(candidate_lines)
        if cleaned:
            return cleaned
    except Exception as e:
        logger.warning(f"Supporting-sentence extraction failed, using deterministic fallback: {str(e)}")

    # Deterministic fallback: sentence split + cleanup
    return clean_candidate_lines(split_sentences(chunk_text))


async def build_dynamic_sources(query: str, answer: str, chunks: List[Dict], max_sources: int = 5) -> List[Dict]:
    """
    Dynamic, query-agnostic source attribution:
    ANSWER -> supporting spans -> filtered evidence
    """
    if not chunks:
        return []

    logger.info(f"Source attribution debug | chunks retrieved: {len(chunks)}")

    evidence = []
    total_extracted = 0
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        metadata = chunk.get("metadata", {}) or {}
        doc_name = chunk.get("doc_name") or metadata.get("doc_name") or metadata.get("document_name") or metadata.get("file_name") or "Unknown"
        page = _to_int(chunk.get("page", metadata.get("page", metadata.get("page_number"))), 1) or 1
        bbox = _parse_bbox(chunk.get("bbox", metadata.get("bbox")))
        base_score = float(chunk.get("rerank_score", chunk.get("score", chunk.get("relevance_score", 0.0))))
        base_score += _section_boost(query, metadata)

        extracted_sentences = await _extract_supporting_sentences(query, answer, text)
        total_extracted += len(extracted_sentences)
        for sentence in extracted_sentences:
            query_sim, answer_sim, combined = score_sentence_relevance(query, answer, sentence)

            # Relevance filtering
            if query_sim <= 0.3:
                continue

            evidence.append({
                "doc": doc_name,
                "file": doc_name,
                "page": page,
                "text": sentence,
                "score": float(max(base_score, combined + _section_boost(query, metadata))),
                "query_similarity": float(query_sim),
                "answer_similarity": float(answer_sim),
                "bbox": bbox,
            })

    logger.info(f"Source attribution debug | sentences extracted: {total_extracted}")
    logger.info(f"Source attribution debug | evidence after filtering: {len(evidence)}")

    if not evidence:
        fallback = _fallback_sources_from_chunks(chunks, limit=2)
        logger.info(f"Source attribution debug | fallback chunk sources used: {len(fallback)}")
        return fallback

    # Deduplicate and rank
    deduped = []
    seen = set()
    for item in sorted(
        evidence,
        key=lambda e: (e.get("score", 0.0), e.get("answer_similarity", 0.0), e.get("query_similarity", 0.0)),
        reverse=True,
    ):
        key = (item.get("doc", ""), item.get("page", 1), _normalize_for_dedupe(item.get("text", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_sources:
            break

    # Always return at least 2 sources by backing off to top chunks if needed
    if len(deduped) < 2:
        fallback = _fallback_sources_from_chunks(chunks, limit=2)
        seen = {(item.get("doc", ""), item.get("page", 1), _normalize_for_dedupe(item.get("text", ""))) for item in deduped}
        for fb in fallback:
            key = (fb.get("doc", ""), fb.get("page", 1), _normalize_for_dedupe(fb.get("text", "")))
            if key in seen:
                continue
            deduped.append(fb)
            seen.add(key)
            if len(deduped) >= 2:
                break

    logger.info(f"Source attribution debug | final sources: {len(deduped)}")

    return deduped


def extract_highlights_from_sources(sources: List[Dict], max_highlights: int = 8) -> List[Dict]:
    """
    Extract exact sentence-level evidence spans from retrieved chunks.

    This is intentionally separate from answer generation.
    """
    if not sources:
        return []

    highlights = []
    for source in sources:
        span_text = (source.get("text") or "").strip()
        if not span_text:
            continue

        page = _to_int(source.get("page"), 1) or 1
        doc_name = source.get("doc") or source.get("file") or "Unknown"
        score = float(source.get("score", 0.0))
        bbox = _parse_bbox(source.get("bbox"))

        highlights.append({
            "doc_name": doc_name,
            "page": page,
            "text": span_text,
            "score": float(score),
            "char_start": None,
            "char_end": None,
            "bbox": bbox,
        })

    # Deduplicate and sort by score
    deduped = []
    seen = set()
    for item in sorted(highlights, key=lambda h: h.get("score", 0.0), reverse=True):
        key = (item["doc_name"], item["page"], item["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= max_highlights:
            break

    return deduped


def _is_page_count_query(question: str) -> bool:
    question = question.lower()
    return (
        "page" in question and
        any(phrase in question for phrase in ["how many", "number of", "total", "pages in", "pages are in"])
    )


def _is_author_query(question: str) -> bool:
    question = question.lower()
    return any(phrase in question for phrase in ["author", "who wrote", "written by", "name of author"])


def _is_title_query(question: str) -> bool:
    question = question.lower()
    return any(phrase in question for phrase in ["title", "name of book", "book name", "document title"])


def _is_summary_query(question: str) -> bool:
    question = question.lower()
    return any(
        phrase in question
        for phrase in [
            "summary",
            "summarize",
            "entire summary",
            "overall summary",
            "summary of the document",
            "summary of this book",
            "what is this document about",
            "what is this book about",
        ]
    )


def _is_word_count_query(question: str) -> bool:
    """Detect queries asking for word count in a document."""
    question = question.lower()
    return (
        "count" in question and
        ("word" in question or "words" in question)
    )


def _page_exists_in_selected_documents(page_number: int, selected_document_ids: List[str]) -> bool:
    """
    Check whether a page number is within range for at least one selected document.
    """
    summaries = get_document_metadata(selected_document_ids)
    for summary in summaries:
        total_pages = _to_int(summary.get("total_pages"), 0) or 0
        if total_pages >= page_number:
            return True
    return False


def _is_direct_image_qa(selected_document_ids: List[str]) -> bool:
    """
    Detect single-image selection and bypass vector similarity retrieval.
    """
    if not selected_document_ids or len(selected_document_ids) != 1:
        return False

    try:
        chunks = get_document_chunks(selected_document_ids, max_chunks_per_document=1)
        if not chunks:
            return False
        metadata = chunks[0].get("metadata", {}) or {}
        return (metadata.get("file_type") or "").lower() == "image"
    except Exception:
        return False


def _validate_indexed_pages(document_id: str, documents: List[Dict]) -> Dict:
    """
    Validate that indexed pages can be retrieved from vector DB by metadata page filter.
    """
    expected_pages = sorted({
        _to_int(doc.get("metadata", {}).get("page", doc.get("metadata", {}).get("page_number")), 0) or 0
        for doc in documents
    })
    expected_pages = [p for p in expected_pages if p > 0]

    pages_with_chunks = []
    pages_without_chunks = []

    for page in expected_pages:
        chunks = get_chunks_by_page([document_id], page_number=page, limit=1)
        if chunks:
            pages_with_chunks.append(page)
        else:
            pages_without_chunks.append(page)

    logger.info(
        f"Index validation | doc_id={document_id} | pages_with_chunks={pages_with_chunks} | "
        f"pages_without_chunks={pages_without_chunks}"
    )

    return {
        "expected_pages": expected_pages,
        "pages_with_chunks": pages_with_chunks,
        "pages_without_chunks": pages_without_chunks,
    }


def classify_query(query: str) -> str:
    """
    Classify query into routing type: count, compare, summary, or qa.
    
    Args:
        query: User's question
        
    Returns:
        Route type: "count", "compare", "summary", or "qa"
    """
    q = query.lower()
    
    # Count queries take priority (most specific)
    if ("count" in q or "number of" in q) and ("word" in q or "words" in q):
        return "count"
    
    # Compare queries
    if any(phrase in q for phrase in ["compare", "difference", "differences", "similarities", "similar", "vs", "versus"]):
        return "compare"
    
    # Summary queries
    if any(phrase in q for phrase in ["summary", "summarize", "summarization", "summarise", "overall", "what is this"]):
        return "summary"
    
    # Default to QA
    return "qa"


def normalize_text(text: str) -> str:
    """
    Normalize text for robust word counting.
    
    - Remove zero-width characters
    - Collapse whitespace
    - Normalize dashes
    
    Args:
        text: Raw text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Remove zero-width characters
    text = text.replace("\u200b", " ")
    
    # Normalize dashes and hyphens
    text = text.replace("\u2013", "-")  # en dash
    text = text.replace("\u2014", "-")  # em dash
    
    # Collapse multiple whitespace into single space
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def count_words(text: str) -> int:
    """
    Count the number of words in text using regex pattern matching.
    Normalizes text first for robustness.
    
    Args:
        text: Text to count words in
        
    Returns:
        Number of words found
    """
    if not text:
        return 0
    
    # Normalize text first
    text = normalize_text(text)
    
    # Match word boundaries: sequences of alphanumeric chars and underscores
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def extract_doc_name(query: str, available_docs: List[str]) -> str | None:
    """
    Extract document name from query by matching against available documents.
    
    Matching strategy:
    1. Exact match (case-insensitive)
    2. Substring match (longest first to avoid partial matches)
    3. Fuzzy match on filename without extension
    
    Args:
        query: User's query
        available_docs: List of available document file names
        
    Returns:
        Matched document file name, or None if no match found
    """
    q = query.lower()
    
    if not available_docs:
        return None
    
    # Strategy 1: Try exact matches with common variations
    for doc_name in available_docs:
        doc_lower = doc_name.lower()
        if doc_lower == q or doc_lower in q:
            return doc_name
    
    # Strategy 2: Try substring matching (longest first to avoid partial matches)
    for doc_name in sorted(available_docs, key=len, reverse=True):
        doc_lower = doc_name.lower()
        if doc_lower in q:
            return doc_name
    
    # Strategy 3: Try matching just the filename without extension
    for doc_name in available_docs:
        name_without_ext = doc_name.rsplit('.', 1)[0].lower()
        if name_without_ext in q:
            return doc_name
    
    return None


def get_full_document_text(document_ids: List[str]) -> Dict[str, str]:
    """
    Retrieve the full text of documents by reconstructing from stored chunks.
    
    Args:
        document_ids: List of document IDs to retrieve
        
    Returns:
        Dictionary mapping document_id to full text
    """
    full_texts = {}
    
    for doc_id in document_ids:
        # Get all chunks for this document, ordered by page and chunk index
        chunks = get_document_chunks([doc_id], max_chunks_per_document=999)
        
        # Reconstruct full text by joining chunks in order
        text_parts = [chunk["text"] for chunk in chunks]
        full_text = " ".join(text_parts)
        full_texts[doc_id] = full_text
    
    return full_texts


def run_count_pipeline(query: str, available_docs: List[str]) -> Dict:
    """
    Deterministic word count pipeline - bypasses LLM, retrieval, and embeddings.
    
    NO LLM CALL - Uses deterministic regex-based word counting
    
    Args:
        query: The user's question
        available_docs: List of available document file names
        
    Returns:
        Dictionary with word count and sources
    """
    try:
        # Handle case where no documents are uploaded
        if not available_docs:
            return {
                "success": False,
                "error": "No documents uploaded",
                "answer": "No documents available. Please upload a document first.",
                "sources": []
            }
        
        # Extract document name from query
        doc_name = extract_doc_name(query, available_docs)
        
        # If no document specified in query, auto-select if only one document exists
        if not doc_name:
            if len(available_docs) == 1:
                doc_name = available_docs[0]
                logger.info(f"Auto-selected single document: {doc_name}")
            else:
                doc_list = ", ".join(available_docs)
                return {
                    "success": False,
                    "error": "Document not specified",
                    "answer": f"Please specify which document to count words for.\n\nAvailable documents:\n- {doc_list}",
                    "sources": []
                }
        
        # Get metadata to find document ID
        from .vectordb import get_all_document_metadata
        metadata_list = get_all_document_metadata()
        
        if not metadata_list:
            return {
                "success": False,
                "error": "No documents indexed",
                "answer": "No documents are currently indexed in the system.",
                "sources": []
            }
        
        doc_id = None
        for metadata in metadata_list:
            if metadata["file_name"] == doc_name:
                doc_id = metadata["document_id"]
                break
        
        if not doc_id:
            return {
                "success": False,
                "error": "Document metadata not found",
                "answer": f"Could not find metadata for {doc_name}. Please check the document name.",
                "sources": []
            }
        
        # Get full document text (reconstructed from chunks)
        full_texts = get_full_document_text([doc_id])
        text = full_texts.get(doc_id, "")
        
        if not text:
            return {
                "success": False,
                "error": "Document text empty",
                "answer": f"Could not retrieve text for {doc_name}",
                "sources": []
            }
        
        # Count words
        word_count = count_words(text)
        
        # Format response
        answer = f"{doc_name} contains {word_count:,} words."
        sources = [{"file": doc_name, "page": 1}]
        
        logger.info(f"Word count: {answer}")
        
        return {
            "success": True,
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        error_msg = f"Error counting words: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "answer": "Unable to count words in document.",
            "sources": []
        }


def _try_answer_metadata_query(question: str, selected_document_ids: List[str]) -> Dict | None:
    """
    Answer simple metadata questions directly from stored document metadata.
    """
    if not selected_document_ids:
        return None

    if not (
        _is_page_count_query(question) or
        _is_author_query(question) or
        _is_title_query(question) or
        _is_word_count_query(question)
    ):
        return None
    
    # Word count query - use deterministic pipeline (no LLM)
    if _is_word_count_query(question):
        return run_count_pipeline(question, selected_document_ids)

    summaries = get_document_metadata(selected_document_ids)
    if not summaries:
        return None

    sources = [
        {
            "file": summary["file_name"],
            "page": summary.get("page_number", 1)
        }
        for summary in summaries
    ]

    if _is_page_count_query(question):
        if len(summaries) == 1:
            summary = summaries[0]
            answer = f"{summary['file_name']} has {summary.get('total_pages', 'an unknown number of')} pages."
        else:
            lines = [f"{summary['file_name']}: {summary.get('total_pages', 'Unknown')} pages" for summary in summaries]
            answer = "Selected documents page counts:\n" + "\n".join(lines)

        return {"success": True, "answer": answer, "sources": sources}

    if _is_author_query(question):
        if len(summaries) == 1:
            summary = summaries[0]
            answer = f"The author of {summary['file_name']} is {summary.get('pdf_author', 'Unknown')}."
        else:
            lines = [f"{summary['file_name']}: {summary.get('pdf_author', 'Unknown')}" for summary in summaries]
            answer = "Selected document authors:\n" + "\n".join(lines)

        return {"success": True, "answer": answer, "sources": sources}

    if _is_title_query(question):
        if len(summaries) == 1:
            summary = summaries[0]
            answer = f"The title is {summary.get('pdf_title', summary['file_name'])}."
        else:
            lines = [f"{summary['file_name']}: {summary.get('pdf_title', summary['file_name'])}" for summary in summaries]
            answer = "Selected document titles:\n" + "\n".join(lines)

        return {"success": True, "answer": answer, "sources": sources}

    return None


async def process_document(file_path: str, file_name: str, document_id: str = None) -> Dict:
    """
    Process a document through the RAG pipeline.
    
    Steps:
    1. Load document
    2. Chunk text (returns LangChain Document objects with enriched metadata)
    3. Generate embeddings
    4. Store in ChromaDB with document_id
    
    Args:
        file_path: Path to document file
        file_name: Name of the file
        document_id: Unique document identifier for persistence
    
    Returns:
        Dictionary with processing results
    """
    try:
        logger.info(f"Processing document: {file_name} (ID: {document_id})")
        
        # Step 1: Load document
        documents = load_document(file_path, file_name)
        logger.info(f"Loaded {len(documents)} pages/sections")

        if documents:
            first_meta = documents[0].get("metadata", {}) or {}
            if (first_meta.get("file_type") or "").lower() == "pdf":
                embedded_images_count = int(first_meta.get("embedded_images_count", 0) or 0)
                page_images_count = int(first_meta.get("page_images_count", 0) or 0)
                fallback_used = bool(first_meta.get("fallback_used", False))
                pages_with_visual_context = first_meta.get("pages_with_visual_context", "")

                logger.info(
                    f"Visual summary | embedded_images={embedded_images_count} | "
                    f"fallback_used={fallback_used} | fallback_page_images={page_images_count} | "
                    f"pages_with_visual_context={pages_with_visual_context or 'none'}"
                )
        
        # Step 2: Chunk documents (returns LangChain Document objects)
        chunks = chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Link page-level document_images to chunk-level image metadata
        document_images = []

        for doc in documents:
            meta = doc.get("metadata", {}) or {}
            imgs = meta.get("document_images", [])

            if imgs:
                document_images.extend(imgs)

        unique = {}
        for img in document_images:
            unique[img["image_path"]] = img

        document_images = list(unique.values())
        print("[DEBUG TOTAL IMAGES]", len(document_images))

        for i, chunk in enumerate(chunks):
            chunk_page = chunk.metadata.get("page")

            matched_images = [
                img["image_path"]
                for img in document_images
                if img.get("page") == chunk_page
            ]

            # 🔥 IMPORTANT: CREATE NEW METADATA OBJECT
            new_metadata = {
                **chunk.metadata,
                "image_paths": "|".join(matched_images),
                "has_visual_context": len(matched_images) > 0
            }

            # 🔥 CRITICAL: REASSIGN BACK TO CHUNK
            chunks[i].metadata = new_metadata

            print(f"[DEBUG] chunk_page={chunk_page}")
            print(f"[DEBUG] matched_images={matched_images}")
            print(f"[DEBUG] has_visual_context={chunks[i].metadata['has_visual_context']}")

        # Create image-only fallback chunks for pages that have images but no text chunks
        image_pages = set()
        for img in document_images:
            img_page = _to_int(img.get("page"), None)
            if img_page and img_page > 0:
                image_pages.add(img_page)

        chunk_pages = set(
            _to_int(c.metadata.get("page", c.metadata.get("page_number")), 0) or 0
            for c in chunks
        )
        chunk_pages = {p for p in chunk_pages if p > 0}

        missing_pages = image_pages - chunk_pages

        base_meta = (documents[0].get("metadata", {}) or {}) if documents else {}
        total_pages = _to_int(base_meta.get("total_pages"), 0) or max(chunk_pages.union(image_pages) or {0})

        for page in sorted(missing_pages):
            print(f"[FIX] Creating image-only chunk for page {page}")

            image_paths = [
                img.get("image_path")
                for img in document_images
                if (_to_int(img.get("page"), None) == page) and img.get("image_path")
            ]
            visual_types = sorted({
                str(img.get("type") or "embedded_image")
                for img in document_images
                if _to_int(img.get("page"), None) == page
            })

            chunk = Document(
                page_content=f"Image content present on page {page}",
                metadata={
                    **base_meta,
                    "page": page,
                    "page_number": page,
                    "total_pages": total_pages,
                    "has_visual_context": True,
                    "image_paths": "|".join(image_paths),
                    "visual_context_types": "|".join(visual_types) if visual_types else "embedded_image",
                    "visual_image_count": len(image_paths),
                    "is_image_only": True,
                }
            )

            chunks.append(chunk)

        pages_with_chunks = set(
            _to_int(c.metadata.get("page", c.metadata.get("page_number")), 0) or 0
            for c in chunks
        )
        pages_with_chunks = {p for p in pages_with_chunks if p > 0}
        print(
            "[FINAL CHECK] Missing pages:",
            set(range(1, total_pages + 1)) - pages_with_chunks
        )

        for c in chunks[:3]:
            print("[AFTER FIX] image_paths:", c.metadata.get("image_paths"))
            print("[AFTER FIX] has_visual_context:", c.metadata.get("has_visual_context"))

        if len(chunks) > 5:
            print("[AFTER FIX]", chunks[5].metadata["image_paths"])
        
        # Add document_id to all chunk metadata
        if document_id:
            for chunk in chunks:
                chunk.metadata["document_id"] = document_id
                chunk.metadata["doc_id"] = document_id

        # Protect image fields without overwriting existing linked values
        for chunk in chunks:
            if "image_paths" not in chunk.metadata:
                chunk.metadata["image_paths"] = ""

            if "has_visual_context" not in chunk.metadata:
                chunk.metadata["has_visual_context"] = False

        # Final metadata verification right before ChromaDB insert
        for c in chunks[:3]:
            print("[FINAL DEBUG] page:", c.metadata.get("page"))
            print("[FINAL DEBUG] image_paths:", c.metadata.get("image_paths"))
            print("[FINAL DEBUG] has_visual_context:", c.metadata.get("has_visual_context"))
        
        # Step 3: Extract data for embeddings and storage
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        print("=== FINAL CHECK BEFORE EMBEDDING ===")
        for c in chunks[:5]:
            print(c.metadata.get("page"), c.metadata.get("image_paths"))
        
        logger.info("Generating embeddings...")
        embeddings = embed_texts(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Step 4: Store in ChromaDB
        document_id = add_documents(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            document_id=document_id
        )
        
        logger.info(f"✓ Document processed successfully: {document_id}")
        logger.info(f"✓ Inserted {len(chunks)} chunks into ChromaDB")
        logger.info(f"✓ Sample metadata: {metadatas[0] if metadatas else 'None'}")

        validation = _validate_indexed_pages(document_id, documents)
        
        return {
            "success": True,
            "document_id": document_id,
            "file_name": file_name,
            "chunks_created": len(chunks),
            "index_validation": validation,
            "message": f"Successfully processed {file_name}"
        }
    
    except Exception as e:
        error_msg = f"Error processing document {file_name}: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "file_name": file_name,
            "error": str(e),
            "message": error_msg
        }


async def rag_query(question: str, top_k: int = 5, selected_document_ids: List[str] = None) -> Dict:
    """
    Execute RAG query pipeline with document filtering.
    
    Steps:
    1. Retrieve relevant chunks from selected documents only
    2. Generate answer using LLM
    3. Return answer with sources
    
    Args:
        question: User's question
        top_k: Number of chunks to retrieve
        selected_document_ids: List of document IDs to filter by (required)
    
    Returns:
        Dictionary with answer and sources
    """
    try:
        logger.info(f"RAG query: {question[:100]}...")
        logger.info(f"Filtering by {len(selected_document_ids) if selected_document_ids else 0} document(s)")
        
        # Validate selected documents
        if not selected_document_ids or len(selected_document_ids) == 0:
            logger.warning("No documents selected for query")
            return {
                "success": False,
                "error": "No documents selected. Please select at least one document.",
                "answer": "Please select at least one document to query.",
                "sources": []
            }

        metadata_answer = _try_answer_metadata_query(question, selected_document_ids)
        if metadata_answer:
            logger.info("Answered query directly from document metadata")
            return metadata_answer

        requested_page = detect_page_query(question)

        if requested_page is not None:
            logger.info(f"Page-specific query detected for page {requested_page}")
            retrieved_chunks = retrieve(
                question,
                top_k=max(10, top_k),
                selected_document_ids=selected_document_ids,
                page_number=requested_page,
            )

            if not retrieved_chunks:
                if _page_exists_in_selected_documents(requested_page, selected_document_ids):
                    return {
                        "success": True,
                        "answer": "Page exists but content could not be extracted properly.",
                        "sources": [],
                        "highlights": [],
                    }

                return {
                    "success": True,
                    "answer": f"No content found for page {requested_page}",
                    "sources": [],
                    "highlights": [],
                }

            result = await generate_answer(question, retrieved_chunks)
            answer = (result.get("answer") or "").strip()
            dynamic_sources = await build_dynamic_sources(question, answer, retrieved_chunks, max_sources=max(3, min(5, top_k)))
            if not dynamic_sources:
                dynamic_sources = _fallback_sources_from_chunks(retrieved_chunks, limit=2)

            result["sources"] = dynamic_sources
            result["highlights"] = extract_highlights_from_sources(dynamic_sources)
            result["success"] = True
            return result

        # Direct image QA path (vision-first): skip semantic similarity retrieval
        if _is_direct_image_qa(selected_document_ids):
            logger.info("Direct image QA path activated (single image document selected)")
            retrieved_chunks = get_document_chunks(
                selected_document_ids,
                max_chunks_per_document=3,
            )

            if not retrieved_chunks:
                return {
                    "success": True,
                    "answer": "No visual information found for the selected image.",
                    "sources": [],
                    "highlights": [],
                }

            result = await generate_answer(question, retrieved_chunks)
            answer = (result.get("answer") or "").strip()
            dynamic_sources = await build_dynamic_sources(question, answer, retrieved_chunks, max_sources=max(3, min(5, top_k)))
            if not dynamic_sources:
                dynamic_sources = _fallback_sources_from_chunks(retrieved_chunks, limit=2)

            result["sources"] = dynamic_sources
            result["highlights"] = extract_highlights_from_sources(dynamic_sources)
            result["success"] = True
            return result

        if _is_summary_query(question):
            logger.info("Summary query detected - using document-wide context")
            retrieved_chunks = get_document_chunks(
                selected_document_ids,
                max_chunks_per_document=30
            )
        else:
            # Step 1: Broad retrieval candidates (query-agnostic, no section filters)
            broad_k = max(10, top_k * 3)
            retrieved_chunks = retrieve(
                question,
                top_k=top_k,
                broad_k=broad_k,
                selected_document_ids=selected_document_ids
            )
            logger.info(f"Source attribution debug | chunks after rerank: {len(retrieved_chunks)}")

        # Step 2: Check if any relevant documents found
        if not retrieved_chunks:
            logger.warning("No relevant documents found for query")
            return {
                "success": True,
                "answer": "No relevant information found in uploaded documents.",
                "sources": [],
                "highlights": []
            }
        
        # Step 3: Generate answer from reranked chunks
        result = await generate_answer(question, retrieved_chunks)
        answer = (result.get("answer") or "").strip()

        # Step 4-9: Dynamic evidence extraction and filtering
        dynamic_sources = await build_dynamic_sources(question, answer, retrieved_chunks, max_sources=max(3, min(5, top_k)))

        # Step 10: Safety fallback if no strong evidence is found
        if not dynamic_sources:
            fallback_sources = _fallback_sources_from_chunks(retrieved_chunks, limit=2)
            return {
                "success": True,
                "answer": answer or "Unable to generate a grounded answer.",
                "sources": fallback_sources,
                "highlights": extract_highlights_from_sources(fallback_sources),
            }

        result["sources"] = dynamic_sources
        result["highlights"] = extract_highlights_from_sources(dynamic_sources)
        result["success"] = True
        
        logger.info("RAG query completed successfully")
        return result
    
    except Exception as e:
        error_msg = f"Error in RAG query: {str(e)}"
        logger.error(error_msg)
        
        # If LLM times out but we have retrieved chunks, return them
        if "timeout" in str(e).lower() and 'retrieved_chunks' in locals() and retrieved_chunks:
            logger.warning("LLM timeout - returning safe fallback")
            sources = extract_sources(retrieved_chunks)

            if _is_summary_query(question):
                return {
                    "success": True,
                    "answer": "The model took too long while preparing a full-document summary. Please try a more specific summary request such as chapter-wise summary, summary of the first half, or summary of a specific topic.",
                    "sources": sources,
                    "highlights": extract_highlights_from_sources(sources)
                }

            return {
                "success": True,
                "answer": "The model took too long to generate a grounded answer. Please try a more specific question.",
                "sources": sources,
                "highlights": extract_highlights_from_sources(sources)
            }
        
        return {
            "success": False,
            "error": error_msg,
            "answer": f"Error processing your question: {str(e)}",
            "sources": [],
            "highlights": []
        }


async def batch_process_documents(file_paths: List[str], file_names: List[str]) -> List[Dict]:
    """
    Process multiple documents in batch.
    
    Args:
        file_paths: List of file paths
        file_names: List of file names
    
    Returns:
        List of processing results
    """
    results = []
    
    for file_path, file_name in zip(file_paths, file_names):
        result = await process_document(file_path, file_name)
        results.append(result)
    
    successful = sum(1 for r in results if r.get("success"))
    logger.info(f"Batch processing complete: {successful}/{len(results)} successful")
    
    return results
