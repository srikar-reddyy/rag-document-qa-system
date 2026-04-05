"""
Document Retriever
Retrieves relevant document chunks for a given query
"""

from typing import List, Dict
import logging
import re
import json
from .embedder import embed_text
from .vectordb import query_documents, get_chunks_by_page, get_document_chunks
from .utils import cosine_similarity

logger = logging.getLogger(__name__)

PAGE_QUERY_REGEX = re.compile(r"\bpage(?:\s+number)?\s*(\d+)\b", re.IGNORECASE)

SECTION_KEYWORDS = {
    "conclusion": [
        "conclusion",
        "conclusions",
        "future work",
        "final remarks",
        "closing remarks",
        "summary",
    ],
    "results": [
        "results",
        "findings",
        "evaluation",
        "experimental results",
        "experiments",
    ],
    "introduction": [
        "introduction",
        "intro",
        "background",
        "overview",
    ],
    "literature review": [
        "literature review",
        "related work",
        "prior work",
        "review of literature",
        "state of the art",
    ],
    "methodology": [
        "methodology",
        "methods",
        "method",
        "approach",
        "experimental setup",
        "materials and methods",
    ],
}


def _normalize_section_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _section_aliases(section: str) -> List[str]:
    aliases = SECTION_KEYWORDS.get(section, [section])
    # Match longest aliases first for stable phrase detection.
    return sorted(aliases, key=len, reverse=True)


def detect_section_query(query: str) -> str | None:
    normalized_query = _normalize_section_text(query)
    if not normalized_query:
        return None

    alias_pairs = []
    for section, aliases in SECTION_KEYWORDS.items():
        for alias in aliases:
            alias_pairs.append((section, alias))

    alias_pairs.sort(key=lambda item: len(item[1]), reverse=True)

    for section, alias in alias_pairs:
        if alias in normalized_query:
            return section

    return None


def _metadata_section(metadata: Dict) -> str:
    normalized = _normalize_section_text(metadata.get("section_title_normalized", ""))
    if normalized and normalized != "unknown":
        return normalized

    return _normalize_section_text(
        metadata.get("section_title") or metadata.get("section") or ""
    )


def _metadata_doc_name(metadata: Dict) -> str:
    return str(
        metadata.get("doc_name")
        or metadata.get("document_name")
        or metadata.get("file_name")
        or "Unknown"
    )


def _metadata_doc_key(metadata: Dict) -> str:
    return str(
        metadata.get("document_id")
        or metadata.get("doc_id")
        or _metadata_doc_name(metadata)
    )


def _chunk_matches_section(chunk: Dict, section: str) -> bool:
    metadata = chunk.get("metadata", {}) or {}
    aliases = _section_aliases(section)

    section_value = _metadata_section(metadata)
    if section_value:
        if section_value == section:
            return True
        for alias in aliases:
            if alias in section_value:
                return True

    text = (chunk.get("text") or "").strip()
    first_line = text.splitlines()[0] if text else ""
    normalized_first_line = _normalize_section_text(first_line)
    for alias in aliases:
        if normalized_first_line == alias:
            return True
        if normalized_first_line.startswith(f"{alias} "):
            return True
        if normalized_first_line.startswith(f"{alias}:"):
            return True
        if normalized_first_line.startswith(f"{alias}-"):
            return True

    return False


def _chunk_section_heading(chunk: Dict) -> str | None:
    metadata = chunk.get("metadata", {}) or {}

    section_value = _metadata_section(metadata)
    if section_value and section_value != "unknown":
        return section_value

    text = (chunk.get("text") or "").strip()
    if not text:
        return None

    first_line = _normalize_section_text(text.splitlines()[0])
    if not first_line:
        return None

    for section, aliases in SECTION_KEYWORDS.items():
        for alias in _section_aliases(section):
            if first_line == alias:
                return section
            if first_line.startswith(f"{alias} "):
                return section
            if first_line.startswith(f"{alias}:"):
                return section
            if first_line.startswith(f"{alias}-"):
                return section

    return None


def _chunk_sort_key(chunk: Dict) -> tuple:
    metadata = chunk.get("metadata", {}) or {}
    return (
        _metadata_doc_key(metadata),
        _to_int(metadata.get("page", metadata.get("page_number")), 1) or 1,
        _to_int(metadata.get("chunk_index"), 0) or 0,
    )


def _build_retrieved_chunk(
    chunk: Dict,
    base_score: float = 1.0,
    mode: str | None = None,
    section: str | None = None,
) -> Dict:
    metadata = chunk.get("metadata", {}) or {}
    doc_text = chunk.get("text", "")
    page = _to_int(metadata.get("page", metadata.get("page_number")), 1) or 1

    payload = {
        "text": doc_text,
        "metadata": metadata,
        "relevance_score": float(base_score),
        "rerank_score": float(base_score),
        "score": float(base_score),
        "chunk_id": metadata.get("chunk_id"),
        "doc_name": _metadata_doc_name(metadata),
        "page": page,
        "char_start": _to_int(metadata.get("char_start")),
        "char_end": _to_int(metadata.get("char_end")),
        "bbox": _parse_bbox(metadata.get("bbox")),
        "words": chunk.get("words") or _parse_words(metadata.get("words_json")),
    }

    if mode:
        payload["mode"] = mode
    if section:
        payload["section"] = section

    return payload


def _collect_contiguous_section_chunks(
    all_chunks: List[Dict],
    section: str,
    max_return: int = 120,
) -> List[Dict]:
    if not all_chunks:
        return []

    ordered_chunks = sorted(all_chunks, key=_chunk_sort_key)
    visited_indices = set()
    section_chunks: List[Dict] = []

    for idx, chunk in enumerate(ordered_chunks):
        if idx in visited_indices:
            continue
        if not _chunk_matches_section(chunk, section):
            continue

        metadata = chunk.get("metadata", {}) or {}
        doc_key = _metadata_doc_key(metadata)

        for cursor in range(idx, len(ordered_chunks)):
            candidate = ordered_chunks[cursor]
            candidate_meta = candidate.get("metadata", {}) or {}

            if _metadata_doc_key(candidate_meta) != doc_key:
                break

            heading = _chunk_section_heading(candidate)
            if cursor > idx and heading and heading != section:
                break

            visited_indices.add(cursor)

            score = 1.1 if heading == section else 1.0
            section_chunks.append(
                _build_retrieved_chunk(
                    candidate,
                    base_score=score,
                    mode="section",
                    section=section,
                )
            )

            if len(section_chunks) >= max_return:
                return section_chunks

    return section_chunks


def _debug_retrieval(query: str, section: str | None, mode: str, chunks: List[Dict]) -> None:
    print("\n[RETRIEVAL DEBUG]")
    print("Query:", query)
    print("Mode:", mode)
    print("Detected Section:", section or "None")

    for chunk in chunks or []:
        print({
            "chunk_id": chunk.get("chunk_id") or (chunk.get("metadata", {}) or {}).get("chunk_id"),
            "section": (chunk.get("metadata", {}) or {}).get("section_title"),
            "preview": (chunk.get("text") or "")[:100],
        })


def extract_page_number(query: str) -> int | None:
    match = re.search(r'page\s*(\d+)', (query or "").lower())
    if match:
        return int(match.group(1))
    return None


def detect_page_query(query: str) -> int | None:
    """
    Detect explicit page-based queries such as:
    - "page 6"
    - "content on page 6"
    - "page number 6"
    """
    if not query:
        return None

    match = PAGE_QUERY_REGEX.search(query)
    if not match:
        return None

    try:
        page_number = int(match.group(1))
        return page_number if page_number > 0 else None
    except Exception:
        return None


def _to_int(value, default: int | None = None):
    try:
        return int(value)
    except Exception:
        return default


def _parse_bbox(value):
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


def _parse_words(value):
    if not value:
        return []

    try:
        entries = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []

    if not isinstance(entries, list):
        return []

    words = []
    for item in entries:
        if not isinstance(item, dict):
            continue

        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue

        try:
            char_start = int(item.get("char_start"))
            char_end = int(item.get("char_end"))
            box = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        except Exception:
            continue

        if char_end <= char_start:
            continue

        words.append({
            "text": str(item.get("text") or ""),
            "char_start": char_start,
            "char_end": char_end,
            "bbox": box,
        })

    return words


def retrieve(
    query: str,
    top_k: int = 5,
    selected_document_ids: List[str] = None,
    broad_k: int | None = None,
    page_number: int | None = None,
) -> List[Dict]:
    """
    Retrieve relevant document chunks for a query.
    
    Args:
        query: User's question
        top_k: Number of chunks to retrieve
        selected_document_ids: List of document IDs to filter by (required)
    
    Returns:
        List of dictionaries with text, metadata, and relevance score
    """
    try:
        # Validate that documents are selected
        if not selected_document_ids or len(selected_document_ids) == 0:
            logger.warning("No documents selected for retrieval")
            return []
        
        logger.info(f"Filtering by {len(selected_document_ids)} selected document(s)")

        detected_section = detect_section_query(query)

        query_page_number = page_number or extract_page_number(query)

        # Page-aware retrieval path (bypass semantic vector search)
        if query_page_number is not None:
            logger.info(f"Page-aware retrieval for page {query_page_number}")
            page_chunks = get_chunks_by_page(
                document_ids=selected_document_ids,
                page_number=query_page_number,
                limit=max(top_k, broad_k or top_k),
            )

            retrieved_chunks = []
            for chunk in page_chunks:
                formatted = _build_retrieved_chunk(chunk, base_score=1.0, mode="page")
                if formatted.get("page") is None:
                    formatted["page"] = query_page_number
                retrieved_chunks.append(formatted)

            if not retrieved_chunks:
                print("[FALLBACK] No chunks for page — retrieving nearby pages")
                nearby_pages = [p for p in [query_page_number - 1, query_page_number + 1] if p > 0]
                for nearby_page in nearby_pages:
                    nearby_chunks = get_chunks_by_page(
                        document_ids=selected_document_ids,
                        page_number=nearby_page,
                        limit=max(1, top_k // 2),
                    )

                    for chunk in nearby_chunks:
                        formatted = _build_retrieved_chunk(chunk, base_score=0.8, mode="page")
                        if formatted.get("page") is None:
                            formatted["page"] = nearby_page
                        retrieved_chunks.append(formatted)

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks via page filter")
            retrieved_chunks = retrieved_chunks[:top_k]
            _debug_retrieval(query, detected_section, mode="page", chunks=retrieved_chunks)
            return retrieved_chunks

        # Section-aware retrieval path (deterministic heading navigation)
        if detected_section:
            print(f"[SECTION QUERY DETECTED] {detected_section}")

            all_chunks = get_document_chunks(
                selected_document_ids,
                max_chunks_per_document=max(1200, top_k * 120),
            )

            section_chunks = _collect_contiguous_section_chunks(
                all_chunks=all_chunks,
                section=detected_section,
                max_return=max(120, top_k * 12),
            )

            if section_chunks:
                print(f"[SECTION MATCH] Returning {len(section_chunks)} section chunk(s)")
                _debug_retrieval(query, detected_section, mode="section", chunks=section_chunks)
                return section_chunks

            print("[SECTION MATCH] No section chunks found - using embedding fallback")

        # Semantic retrieval path
        logger.info(f"Generating embedding for query: {query[:100]}...")
        query_embedding = embed_text(query)

        candidate_k = broad_k if broad_k is not None else max(10, top_k * 3)
        
        # Query ChromaDB with document filtering
        results = query_documents(
            query_embedding, 
            top_k=candidate_k,
            document_ids=selected_document_ids
        )
        
        logger.info(f"✓ ChromaDB query completed")
        logger.info(f"  - Documents found: {len(results.get('documents', []))}")
        logger.info(f"  - Distances: {results.get('distances', [])}")
        
        # Handle empty results
        if not results["documents"]:
            logger.warning("❌ No relevant documents found in ChromaDB")
            logger.warning(f"  - Selected document IDs: {selected_document_ids}")
            logger.warning(f"  - Query: {query[:100]}...")
            _debug_retrieval(query, detected_section, mode="semantic-empty", chunks=[])
            return []
        
        # Format results (broad candidates)
        retrieved_chunks = []
        for doc, metadata, distance in zip(
            results["documents"],
            results["metadatas"],
            results["distances"]
        ):
            page = _to_int(metadata.get("page", metadata.get("page_number")), 1) or 1
            score = max(0.0, min(1.0, 1 - distance))
            retrieved_chunks.append({
                "text": doc,
                "metadata": metadata,
                "relevance_score": score,  # ANN similarity
                "score": score,
                "chunk_id": metadata.get("chunk_id"),
                "doc_name": metadata.get("doc_name", metadata.get("document_name", metadata.get("file_name", "Unknown"))),
                "page": page,
                "char_start": _to_int(metadata.get("char_start")),
                "char_end": _to_int(metadata.get("char_end")),
                "bbox": _parse_bbox(metadata.get("bbox")),
                "words": _parse_words(metadata.get("words_json")),
            })

        page_num = extract_page_number(query)

        if page_num:
            print(f"[DEBUG] Page-specific query detected: {page_num}")

            page_chunks = [
                c for c in retrieved_chunks
                if c.get("metadata", {}).get("page") == page_num
            ]

            if page_chunks:
                print(f"[DEBUG] Using {len(page_chunks)} chunks from page {page_num}")
                retrieved_chunks = page_chunks
            else:
                print("[DEBUG] No exact page match, using semantic search fallback")
                retrieved_chunks = sorted(
                    retrieved_chunks,
                    key=lambda c: c.get("metadata", {}).get("page") == page_num,
                    reverse=True
                )

        # Dynamic reranking stage (query-to-chunk semantic relevance)
        for chunk in retrieved_chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                chunk["rerank_score"] = 0.0
                continue

            chunk_embedding = embed_text(text)
            rerank_score = cosine_similarity(query_embedding, chunk_embedding)
            chunk["rerank_score"] = rerank_score
            # Promote rerank score as primary relevance for downstream filtering/sorting
            chunk["score"] = max(chunk.get("score", 0.0), rerank_score)

        retrieved_chunks.sort(
            key=lambda c: (c.get("rerank_score", 0.0), c.get("relevance_score", 0.0)),
            reverse=True
        )
        retrieved_chunks = retrieved_chunks[:top_k]
        
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        _debug_retrieval(query, detected_section, mode="semantic", chunks=retrieved_chunks)
        return retrieved_chunks
    
    except Exception as e:
        logger.exception(f"Error retrieving documents: {str(e)}")
        return []


def format_retrieved_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks with entity-aware attribution.
    
    Ensures that extracted entities (persons, organizations, roles) are prominently
    displayed to prevent misattribution in LLM responses. Uses entity extraction
    results to provide accurate context.
    
    Args:
        chunks: Retrieved document chunks with entity metadata
    
    Returns:
        Formatted context string with entity attribution
    """
    if not chunks:
        return ""
    
    context_parts = []
    current_document = None
    
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        file_name = metadata.get("file_name", "Unknown")
        page_number = metadata.get("page_number", "N/A")
        pdf_author = metadata.get("pdf_author")
        pdf_title = metadata.get("pdf_title")
        section_title = metadata.get("section_title", "Unknown Section")
        
        # Extract entity metadata (pipe-separated strings for ChromaDB compatibility)
        primary_entity = metadata.get("primary_entity", "Unknown")
        
        # Parse pipe-separated entity strings back into lists
        entity_persons_str = metadata.get("entity_persons", "")
        entity_organizations_str = metadata.get("entity_organizations", "")
        
        entity_persons = [p.strip() for p in entity_persons_str.split("|") if p.strip()] if entity_persons_str else []
        entity_organizations = [o.strip() for o in entity_organizations_str.split("|") if o.strip()] if entity_organizations_str else []
        
        text = chunk["text"]

        # Add document separator when switching to a new document
        # CRITICAL: Put primary entity first to ensure attribution
        if current_document != file_name:
            current_document = file_name
            separator = f"\n{'='*80}\n"
            separator += f"PRIMARY SUBJECT: {primary_entity}\n"
            if entity_persons:
                separator += f"PEOPLE: {', '.join(entity_persons)}\n"
            if entity_organizations:
                separator += f"ORGANIZATIONS: {', '.join(entity_organizations)}\n"
            separator += f"SOURCE: {file_name}\n"
            if pdf_title:
                separator += f"TITLE: {pdf_title}\n"
            separator += f"{'='*80}\n"
            context_parts.append(separator)

        header_parts = [f"Page: {page_number}", f"Section: {section_title}"]
        context_parts.append(
            f"[{', '.join(header_parts)}]\n{text}\n"
        )
    
    return "\n---\n".join(context_parts)


def extract_sources(chunks: List[Dict]) -> List[Dict]:
    """
    Extract unique source references from retrieved chunks.
    
    Args:
        chunks: Retrieved document chunks
    
    Returns:
        List of unique source dictionaries
    """
    sources = []
    seen = set()
    
    for chunk in chunks:
        metadata = chunk["metadata"]
        file_name = metadata.get("file_name", "Unknown")
        page_number = _to_int(metadata.get("page", metadata.get("page_number")), 1) or 1
        
        # Create unique identifier
        source_key = f"{file_name}_{page_number}"
        
        if source_key not in seen:
            sources.append({
                "file": file_name,
                "page": page_number
            })
            seen.add(source_key)
    
    return sources
