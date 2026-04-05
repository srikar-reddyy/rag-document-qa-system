"""
RAG utility helpers for evidence extraction and filtering.
"""

from __future__ import annotations

from typing import List, Tuple
import math
import re

from .embedder import embed_text


HEADING_PATTERNS = [
    r"^[A-Z][A-Z\s&/-]{2,}$",
    r"^(subject|source|title|document|page|section)\s*:\s*.*$",
]

PUBLICATION_NOISE_PATTERNS = [
    r"\bdoi\s*:",
    r"\bissn\b",
    r"\bwww\.",
    r"\b\d+\s*\|\s*page\b",
    r"\bdate\s+of\s+submission\b",
    r"\bdate\s+of\s+acceptance\b",
]


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def is_noisy_line(line: str) -> bool:
    if not line:
        return True

    value = line.strip()
    if len(value) < 20:
        return True

    for pattern in HEADING_PATTERNS:
        if re.match(pattern, value, re.IGNORECASE):
            return True

    for pattern in PUBLICATION_NOISE_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            return True

    # OCR fragments with very low alphabetic ratio
    alpha = sum(1 for ch in value if ch.isalpha())
    if alpha / max(1, len(value)) < 0.45:
        return True

    return False


def clean_candidate_lines(lines: List[str]) -> List[str]:
    cleaned = []
    seen = set()

    for line in lines:
        line = line.strip(" -•\t")
        if is_noisy_line(line):
            continue
        key = normalize_text(line)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(line)

    return cleaned


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def token_overlap_ratio(text_a: str, text_b: str) -> float:
    a = set(re.findall(r"\w+", (text_a or "").lower()))
    b = set(re.findall(r"\w+", (text_b or "").lower()))
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def score_sentence_relevance(query: str, answer: str, sentence: str) -> Tuple[float, float, float]:
    """
    Returns tuple: (query_similarity, answer_similarity, combined_score)
    """
    q_emb = embed_text(query)
    a_emb = embed_text(answer)
    s_emb = embed_text(sentence)

    query_similarity = cosine_similarity(q_emb, s_emb)
    answer_similarity = cosine_similarity(a_emb, s_emb)

    overlap = token_overlap_ratio(answer, sentence)
    combined = (0.55 * query_similarity) + (0.35 * answer_similarity) + (0.10 * overlap)

    return query_similarity, answer_similarity, combined
