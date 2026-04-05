"""
Document Chunker
Semantic, adaptive, sentence-based chunking for improved RAG retrieval quality.
"""

from typing import List, Dict, Any
import logging
import importlib
import re
import json
from dataclasses import dataclass

from .embedder import embed_text
from .utils import cosine_similarity

try:
    _blingfire_mod = importlib.import_module("blingfire")
    text_to_sentences = getattr(_blingfire_mod, "text_to_sentences")
except Exception:
    text_to_sentences = None

try:
    _transformers_mod = importlib.import_module("transformers")
    AutoTokenizer = getattr(_transformers_mod, "AutoTokenizer")
except Exception:
    AutoTokenizer = None


# Load Document class dynamically to avoid hard dependency breakages
try:
    _lc_docs_mod = importlib.import_module("langchain_core.documents")
    Document = getattr(_lc_docs_mod, "Document")
except Exception:
    @dataclass
    class Document:
        page_content: str
        metadata: dict

logger = logging.getLogger(__name__)

# Adaptive chunk size defaults (in actual tokens)
DEFAULT_CHUNK = 400
LARGE_CHUNK = 700
SMALL_CHUNK = 250

# Sentence overlap (not char overlap)
DEFAULT_OVERLAP_SENTENCES = 1
MAX_OVERLAP_SENTENCES = 2

# Quality filters
MIN_CHUNK_TOKENS = 20
DEDUP_SIMILARITY_THRESHOLD = 0.90

SECTION_KEYWORDS = {
    "conclusion": [
        "conclusion",
        "conclusions",
        "future work",
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

_tokenizer = None


def get_tokenizer():
    """
    Lazily initialize tokenizer used by embedding model family for accurate token counts.
    """
    global _tokenizer
    if _tokenizer is None:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is not installed. Install it to enable token-accurate chunking.")
        _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return _tokenizer


def count_tokens(text: str) -> int:
    """
    Count tokens using the real tokenizer (not char approximation).
    """
    if not text or not text.strip():
        return 0
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _to_int(value, default: int = 0) -> int:
    """Best-effort conversion to int."""
    try:
        return int(value)
    except Exception:
        return default


def split_sentences(text: str) -> List[str]:
    """
    Robust sentence splitter. Uses BlingFire when available, regex fallback otherwise.
    """
    if not text:
        return []

    if text_to_sentences is not None:
        try:
            return [s.strip() for s in text_to_sentences(text).split("\n") if s and s.strip()]
        except Exception:
            pass

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in parts if s and s.strip()]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _is_broken_ocr_fragment(line: str) -> bool:
    """
    Detect likely OCR garbage lines/fragments.
    """
    value = (line or "").strip()
    if not value:
        return True

    # Mostly symbols/digits with weak alphabetic signal
    alpha = sum(1 for ch in value if ch.isalpha())
    alnum = sum(1 for ch in value if ch.isalnum())
    if alnum == 0:
        return True

    if alpha / max(1, len(value)) < 0.25:
        return True

    # Extremely long token without spaces is often OCR artifact
    if len(value) > 40 and " " not in value:
        return True

    return False


def clean_chunk_text(text: str) -> str:
    """
    Clean chunk text by removing empty/noisy lines and normalizing whitespace.
    """
    lines = [ln.strip() for ln in (text or "").splitlines()]
    kept = [ln for ln in lines if ln and not _is_broken_ocr_fragment(ln)]
    return _normalize_whitespace(" ".join(kept))


def infer_adaptive_chunk_size(text: str, metadata: Dict[str, Any]) -> int:
    """
    Adaptive chunk sizing heuristic:
    - structured docs (resume/table-like): SMALL_CHUNK
    - long narrative docs: LARGE_CHUNK
    - default: DEFAULT_CHUNK
    """
    raw = text or ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    lowered_name = str(metadata.get("file_name", "")).lower()
    section = str(metadata.get("section_title", "")).lower()

    bullet_like = sum(1 for ln in lines if re.match(r"^[-•*]\s+", ln))
    colon_like = sum(1 for ln in lines if ":" in ln)
    short_lines = sum(1 for ln in lines if len(ln.split()) <= 8)
    table_like_ratio = (bullet_like + colon_like + short_lines) / max(1, len(lines))

    # Narrative signal
    sentences = split_sentences(raw)
    avg_sentence_tokens = 0.0
    if sentences:
        avg_sentence_tokens = sum(max(1, count_tokens(s)) for s in sentences) / len(sentences)

    structured_hint = any(key in lowered_name for key in ["resume", "cv"]) or any(
        key in section for key in ["skills", "experience", "education", "projects"]
    )

    if structured_hint or table_like_ratio >= 0.55:
        return SMALL_CHUNK
    if len(raw) > 6000 and avg_sentence_tokens >= 18:
        return LARGE_CHUNK
    return DEFAULT_CHUNK


def _extract_keywords(text: str, top_k: int = 20) -> List[str]:
    tokens = re.findall(r"\b[a-zA-Z]{4,}\b", (text or "").lower())
    if not tokens:
        return []

    stop = {
        "this", "that", "with", "from", "have", "were", "your", "their", "about", "into", "which",
        "will", "would", "could", "should", "there", "where", "when", "what", "been", "being", "than",
        "then", "them", "they", "also", "such", "using", "used", "between", "after", "before", "over",
        "under", "into", "across", "more", "most", "other", "some", "many", "much"
    }

    freq = {}
    for tok in tokens:
        if tok in stop:
            continue
        freq[tok] = freq.get(tok, 0) + 1

    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in ranked[:top_k]]


def _entity_density(chunk: str, metadata: Dict[str, Any]) -> float:
    text = (chunk or "").lower()
    if not text:
        return 0.0

    entities = []
    for field in ["entity_persons", "entity_organizations", "entity_roles"]:
        value = metadata.get(field, "")
        if isinstance(value, str):
            entities.extend([item.strip().lower() for item in value.split("|") if item.strip()])
        elif isinstance(value, list):
            entities.extend([str(item).strip().lower() for item in value if str(item).strip()])

    if not entities:
        return 0.0

    hits = sum(1 for ent in entities if ent and ent in text)
    return hits / max(1, len(entities))


def _section_importance(section_title: str) -> float:
    section = (section_title or "").strip().lower()
    if not section or section == "unknown":
        return 0.2

    important = {
        "summary": 1.0,
        "abstract": 1.0,
        "experience": 0.9,
        "methodology": 0.85,
        "results": 0.9,
        "conclusion": 0.8,
        "skills": 0.8,
        "projects": 0.8,
        "education": 0.7,
    }
    for key, value in important.items():
        if key in section:
            return value
    return 0.5


def compute_semantic_density(chunk: str, source_keywords: List[str], metadata: Dict[str, Any]) -> float:
    """
    Query-agnostic chunk quality score combining keyword/entity/section signals.
    """
    lowered = (chunk or "").lower()
    if not lowered:
        return 0.0

    words = re.findall(r"\b\w+\b", lowered)
    keyword_hits = sum(1 for kw in source_keywords if kw in lowered)
    keyword_density = keyword_hits / max(1, len(words))

    entity_density = _entity_density(chunk, metadata)
    section_score = _section_importance(str(metadata.get("section_title", "Unknown")))

    # Weighted bounded score in [0, 1]
    score = (0.45 * min(1.0, keyword_density * 12.0)) + (0.35 * entity_density) + (0.20 * section_score)
    return max(0.0, min(1.0, score))


def _apply_sentence_overlap(chunk_sentences: List[List[str]], overlap_sentences: int) -> List[str]:
    """
    Sentence-based overlap to preserve context continuity.
    """
    merged = []
    for i, sentences in enumerate(chunk_sentences):
        current = list(sentences)
        if i > 0 and overlap_sentences > 0:
            prev_tail = chunk_sentences[i - 1][-overlap_sentences:]
            current = prev_tail + current
        merged.append(_normalize_whitespace(" ".join(current)))
    return merged


def _dedupe_chunks_semantic(chunks: List[str]) -> List[str]:
    """
    Deduplicate near-identical consecutive chunks using embedding cosine similarity.
    """
    deduped: List[str] = []
    prev_embedding = None

    for chunk in chunks:
        if not chunk:
            continue

        try:
            emb = embed_text(chunk)
        except Exception as e:
            logger.warning(f"Chunk dedupe embedding failed, falling back to lexical check: {str(e)}")
            emb = None

        if deduped:
            if emb is not None and prev_embedding is not None:
                sim = cosine_similarity(emb, prev_embedding)
                if sim >= DEDUP_SIMILARITY_THRESHOLD:
                    continue
            else:
                if chunk.lower() == deduped[-1].lower():
                    continue

        deduped.append(chunk)
        prev_embedding = emb

    return deduped


def _group_sentences_semantic(sentences: List[str], chunk_size_tokens: int) -> List[List[str]]:
    """
    Group sentences into chunks capped by true token count.
    """
    groups: List[List[str]] = []
    current_group: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent = _normalize_whitespace(sentence)
        if not sent:
            continue

        sentence_tokens = count_tokens(sent)
        if sentence_tokens == 0:
            continue

        # Hard cap very long sentence by itself
        if sentence_tokens >= chunk_size_tokens:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_tokens = 0
            groups.append([sent])
            continue

        if current_group and current_tokens + sentence_tokens > chunk_size_tokens:
            groups.append(current_group)
            current_group = [sent]
            current_tokens = sentence_tokens
        else:
            current_group.append(sent)
            current_tokens += sentence_tokens

    if current_group:
        groups.append(current_group)

    return groups


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    metadata: Dict[str, Any] | None = None,
) -> List[str]:
    """
    Split text into semantic chunks using sentence grouping and true token counts.
    """
    if not text or not text.strip():
        return []

    meta = metadata or {}
    target_chunk_size = chunk_size or infer_adaptive_chunk_size(text, meta)
    overlap_sentences = DEFAULT_OVERLAP_SENTENCES
    if overlap is not None:
        overlap_sentences = max(0, min(MAX_OVERLAP_SENTENCES, overlap))

    sentences = split_sentences(text)
    groups = _group_sentences_semantic(sentences, target_chunk_size)
    with_overlap = _apply_sentence_overlap(groups, overlap_sentences=overlap_sentences)

    cleaned = []
    for ch in with_overlap:
        value = clean_chunk_text(ch)
        if not value:
            continue
        if count_tokens(value) < MIN_CHUNK_TOKENS:
            continue
        cleaned.append(value)

    return _dedupe_chunks_semantic(cleaned)


def detect_section_title(text: str) -> str:
    """
    Detect section title from the beginning of a text chunk.
    
    Looks for common patterns like:
    - All caps lines (INTRODUCTION)
    - Numbered sections (1. Introduction, 1.1 Background)
    - Title case headings followed by newline
    
    Args:
        text: Text chunk to analyze
    
    Returns:
        Detected section title or "Unknown"
    """
    lines = [line.strip() for line in (text or "").split("\n") if line and line.strip()]
    if not lines:
        return "Unknown"

    heading_lines = lines[:5]
    for line in heading_lines:
        candidate = re.sub(r"^(\d+\.)+\s*", "", line).strip(" :-\t")
        if not candidate:
            continue

        lowered = candidate.lower()

        for canonical, aliases in SECTION_KEYWORDS.items():
            for alias in aliases:
                if lowered == alias:
                    return canonical.title()
                if lowered.startswith(f"{alias} ") or lowered.startswith(f"{alias}:") or lowered.startswith(f"{alias}-"):
                    return canonical.title()

        if re.match(r"^(\d+\.)+\s*[A-Za-z][A-Za-z\s\-]{1,80}$", line):
            return candidate

        if candidate.isupper() and len(candidate.split()) <= 8:
            return candidate

    return "Unknown"


def normalize_section_title(section_title: str) -> str:
    normalized = _normalize_whitespace(section_title).lower()
    if not normalized or normalized == "unknown":
        return "unknown"

    for canonical, aliases in SECTION_KEYWORDS.items():
        if any(alias in normalized for alias in aliases):
            return canonical

    return normalized


def _parse_page_words(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load page-level word annotations from loader metadata."""
    raw = metadata.get("page_words_json")
    if not raw:
        return []

    try:
        entries = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []

    if not isinstance(entries, list):
        return []

    words: List[Dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text") or "")
        char_start = _to_int(item.get("char_start"), -1)
        char_end = _to_int(item.get("char_end"), -1)
        bbox = item.get("bbox")

        if not text or char_start < 0 or char_end <= char_start:
            continue

        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue

        try:
            box = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        except Exception:
            continue

        words.append({
            "text": text,
            "char_start": char_start,
            "char_end": char_end,
            "bbox": box,
        })

    words.sort(key=lambda w: int(w.get("char_start", 0)))
    return words


def _slice_chunk_words(
    page_words: List[Dict[str, Any]],
    chunk_char_start: int,
    chunk_char_end: int,
    chunk_text: str,
) -> List[Dict[str, Any]]:
    """Build chunk-local word annotations from page-level words."""
    if not page_words or chunk_char_end <= chunk_char_start:
        return []

    out: List[Dict[str, Any]] = []
    chunk_len = len(chunk_text or "")

    for word in page_words:
        ws = _to_int(word.get("char_start"), -1)
        we = _to_int(word.get("char_end"), -1)
        if ws < chunk_char_start or we > chunk_char_end:
            continue

        rel_start = max(0, ws - chunk_char_start)
        rel_end = min(chunk_len, max(rel_start, we - chunk_char_start))
        if rel_end <= rel_start:
            continue

        word_text = (chunk_text or "")[rel_start:rel_end] or str(word.get("text") or "")
        out.append({
            "text": word_text,
            "char_start": rel_start,
            "char_end": rel_end,
            "bbox": word.get("bbox"),
        })

    return out


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[Document]:
    """
    Chunk multiple documents using semantic paragraph-based chunking.
    
    This function uses LangChain's RecursiveCharacterTextSplitter to create
    semantically coherent chunks that preserve meaning and improve retrieval
    quality for multi-document reasoning.
    
    Each chunk includes enriched metadata:
    - document_name: Source file name
    - page_number: Page where chunk originated
    - section_title: Detected section heading (or "Unknown")
    - chunk_index: Position within document
    - total_chunks: Total chunks from source
    - chunk_size_tokens: Approximate token count
    
    Args:
        documents: List of documents with text and metadata (dict format)
        chunk_size: Target chunk size in tokens (default: 400)
        overlap: Overlap between chunks in tokens (default: 80)
    
    Returns:
        List of LangChain Document objects with enriched metadata
    """
    all_chunked_docs = []
    active_section_by_document: Dict[str, str] = {}
    total_sentences = 0
    total_tokens = 0
    
    for doc in documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        page_number = _to_int(metadata.get("page", metadata.get("page_number", 1)), 1)
        adaptive_size = chunk_size or infer_adaptive_chunk_size(text, metadata)
        page_words = _parse_page_words(metadata)
        document_name = metadata.get("file_name", metadata.get("document_name", "unknown"))
        
        if not text or not text.strip():
            logger.warning(
                f"Page {page_number} → length: 0 → chunks: 0 (empty extracted text)"
            )
            continue

        sentences = split_sentences(text)
        total_sentences += len(sentences)

        chunk_texts = chunk_text(
            text,
            chunk_size=adaptive_size,
            overlap=overlap,
            metadata=metadata,
        )
        total_chunks = len(chunk_texts)
        cursor = 0
        source_keywords = _extract_keywords(text)
        
        # Enrich each chunk with enhanced metadata
        for chunk_idx, chunk_body in enumerate(chunk_texts):
            # Compute chunk char offsets in source page text
            search_start = max(0, cursor - 200)
            char_start = text.find(chunk_body, search_start)
            if char_start == -1:
                char_start = text.find(chunk_body)
            if char_start == -1:
                char_start = cursor
            char_end = min(len(text), char_start + len(chunk_body))
            cursor = max(cursor, char_end)

            # Detect section title from chunk content, then propagate active section
            # so continuation chunks inherit the section heading.
            detected_section_title = detect_section_title(chunk_body)
            if detected_section_title != "Unknown":
                active_section_by_document[document_name] = detected_section_title
                section_title = detected_section_title
            else:
                section_title = active_section_by_document.get(document_name, "Unknown")

            section_title_normalized = normalize_section_title(section_title)

            token_count = count_tokens(chunk_body)
            sentence_count = len(split_sentences(chunk_body))
            semantic_density = compute_semantic_density(
                chunk_body,
                source_keywords=source_keywords,
                metadata={**metadata, "section_title": section_title},
            )
            total_tokens += token_count
            
            # Get entity data from source metadata
            entity_persons = metadata.get("entity_persons", [])
            entity_organizations = metadata.get("entity_organizations", [])
            entity_roles = metadata.get("entity_roles", [])
            
            # Convert lists to pipe-separated strings for ChromaDB compatibility
            # (ChromaDB doesn't accept lists in metadata, only primitives)
            persons_str = "|".join(entity_persons) if entity_persons else ""
            orgs_str = "|".join(entity_organizations) if entity_organizations else ""
            roles_str = "|".join(entity_roles) if entity_roles else ""
            
            # Enrich metadata - use only flat, primitive types for ChromaDB compatibility
            chunk_metadata = dict(metadata)
            chunk_id = f"{document_name}::p{page_number}::c{chunk_idx}"
            chunk_words = _slice_chunk_words(page_words, char_start, char_end, chunk_body)

            chunk_metadata.update({
                "doc_name": document_name,
                "document_name": document_name,
                "chunk_id": chunk_id,
                "page": page_number,
                "page_number": page_number,
                "section_title": section_title,
                "section_title_normalized": section_title_normalized,
                "chunk_index": chunk_idx,
                "total_chunks": total_chunks,
                "chunk_size_tokens": token_count,
                "char_start": char_start,
                "char_end": char_end,
                "sentence_count": sentence_count,
                "token_count": token_count,
                "semantic_density": float(semantic_density),
                "bbox": metadata.get("bbox", ""),
                "words_json": json.dumps(chunk_words, ensure_ascii=False),
                "words_count": len(chunk_words),
                # Entity extraction data (flattened for ChromaDB)
                "primary_entity": metadata.get("primary_entity", "Unknown"),
                "entity_persons": persons_str,
                "entity_organizations": orgs_str,
                "entity_roles": roles_str
            })

            # Keep chunk-level metadata compact and avoid repeating page-level words.
            chunk_metadata.pop("page_words_json", None)

            chunk = Document(
                page_content=chunk_body,
                metadata=chunk_metadata
            )
            
            all_chunked_docs.append(chunk)

        logger.info(
            f"Page {page_number} → length: {len(text.strip())} → chunks: {total_chunks} → target_chunk_size: {adaptive_size}"
        )
    
    logger.info(f"Created {len(all_chunked_docs)} semantic chunks from {len(documents)} documents")
    avg_tokens = (total_tokens / len(all_chunked_docs)) if all_chunked_docs else 0.0
    logger.info(f"Chunking stats → total_sentences: {total_sentences}, total_chunks: {len(all_chunked_docs)}, avg_tokens_per_chunk: {avg_tokens:.2f}")
    
    # Log sample metadata for verification
    if all_chunked_docs:
        sample_meta = all_chunked_docs[0].metadata
        logger.info(f"Sample chunk metadata: {sample_meta}")
    
    return all_chunked_docs
