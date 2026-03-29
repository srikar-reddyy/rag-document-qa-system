"""
Document Retriever
Retrieves relevant document chunks for a given query
"""

from typing import List, Dict
import logging
import re
from .embedder import embed_text
from .vectordb import query_documents, get_chunks_by_page
from .utils import cosine_similarity

logger = logging.getLogger(__name__)

PAGE_QUERY_REGEX = re.compile(r"\bpage(?:\s+number)?\s*(\d+)\b", re.IGNORECASE)


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

        # Page-aware retrieval path (bypass semantic vector search)
        if page_number is not None:
            logger.info(f"Page-aware retrieval for page {page_number}")
            page_chunks = get_chunks_by_page(
                document_ids=selected_document_ids,
                page_number=page_number,
                limit=max(top_k, broad_k or top_k),
            )

            retrieved_chunks = []
            for chunk in page_chunks:
                metadata = chunk.get("metadata", {})
                doc_text = chunk.get("text", "")
                page = _to_int(metadata.get("page", metadata.get("page_number")), page_number) or page_number
                retrieved_chunks.append({
                    "text": doc_text,
                    "metadata": metadata,
                    "relevance_score": 1.0,
                    "rerank_score": 1.0,
                    "score": 1.0,
                    "doc_name": metadata.get("doc_name", metadata.get("document_name", metadata.get("file_name", "Unknown"))),
                    "page": page,
                    "char_start": _to_int(metadata.get("char_start")),
                    "char_end": _to_int(metadata.get("char_end")),
                    "bbox": _parse_bbox(metadata.get("bbox")),
                })

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks via page filter")
            return retrieved_chunks[:top_k]

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
                "doc_name": metadata.get("doc_name", metadata.get("document_name", metadata.get("file_name", "Unknown"))),
                "page": page,
                "char_start": _to_int(metadata.get("char_start")),
                "char_end": _to_int(metadata.get("char_end")),
                "bbox": _parse_bbox(metadata.get("bbox")),
            })

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
