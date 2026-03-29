"""
Answer Generator
Constructs prompts and calls OpenRouter Chat Completions API
"""

import httpx
import os
from typing import List, Dict
import logging
import json
from typing import AsyncIterator
import re

logger = logging.getLogger(__name__)

# LLM API Configuration (OpenRouter)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT = 45.0


def _extract_requested_point_count(query: str) -> int | None:
    """
    Try to detect explicit list-size intent, e.g.:
    - "give me 10 points"
    - "top 7 reasons"
    - "list 5 items"
    """
    q = (query or "").lower()

    patterns = [
        r"\b(?:top|list|give|provide|share|tell)\s+(\d{1,2})\b",
        r"\b(\d{1,2})\s+(?:key\s+)?(?:points?|items?|reasons?|insights?|takeaways?|facts?|differences?)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            try:
                n = int(match.group(1))
                if 1 <= n <= 20:
                    return n
            except Exception:
                continue

    return None


def _extract_numbered_points(answer: str) -> List[tuple[int, str]]:
    """
    Parse numbered points in the form:
    1. **Title**\nBody...
    """
    if not answer:
        return []

    pattern = re.compile(r"(?ms)^\s*(\d+)\.\s+(.*?)(?=^\s*\d+\.\s+|\Z)")
    parsed: List[tuple[int, str]] = []
    for m in pattern.finditer(answer):
        try:
            idx = int(m.group(1))
        except Exception:
            continue
        body = (m.group(2) or "").strip()
        if body:
            parsed.append((idx, body))
    return parsed


def _is_point_complete(text: str) -> bool:
    if not text:
        return False

    stripped = text.strip()
    if len(stripped) < 12:
        return False

    # Must end with sentence punctuation to avoid abrupt cuts.
    return bool(re.search(r"[.!?]\s*$", stripped))


def _validate_numbered_list(answer: str, n_points: int) -> tuple[bool, str]:
    if not answer or not n_points:
        return False, "empty_answer"

    parsed = _extract_numbered_points(answer)
    if len(parsed) != n_points:
        return False, f"count_mismatch:{len(parsed)}"

    expected = list(range(1, n_points + 1))
    got = [idx for idx, _ in parsed]
    if got != expected:
        return False, "index_sequence_invalid"

    for _, point_body in parsed:
        if not _is_point_complete(point_body):
            return False, "incomplete_point"

    # Global anti-truncation sanity.
    if not _is_point_complete(answer):
        return False, "truncated_tail"

    return True, "ok"


async def _continue_missing_points(
    query: str,
    context: str,
    partial_answer: str,
    requested_points: int,
    model_name: str,
) -> str:
    """
    If model stopped early, request only the remaining numbered points.
    """
    parsed = _extract_numbered_points(partial_answer)
    count = len(parsed)
    if count <= 0 or count >= requested_points:
        return partial_answer

    next_idx = count + 1
    continuation_prompt = f"""
Continue the numbered list below.

MANDATORY:
- Output ONLY the remaining points from {next_idx} to {requested_points}.
- Keep exact numbering.
- Each point must be complete (1-3 sentences) and end with punctuation.
- Do not repeat previous points.
- If context is limited, include: "Based on limited available context." in affected points.

User query:
{query}

Document excerpts:
{context}

Already generated points:
{partial_answer}
""".strip()

    continuation = await call_llm_api(
        continuation_prompt,
        model=model_name,
        max_tokens=780,
        temperature=0.2,
    )

    merged = f"{partial_answer.strip()}\n\n{continuation.strip()}".strip()
    return merged


def _select_model_for_query(query: str, retrieved_chunks: List[Dict]) -> str:
    q = (query or "").lower()
    image_query = any(k in q for k in ["image", "figure", "diagram", "photo", "screenshot"])
    has_image_chunks = any(
        (chunk.get("metadata", {}) or {}).get("file_type") == "image"
        or (chunk.get("metadata", {}) or {}).get("has_visual_context")
        for chunk in (retrieved_chunks or [])
    )
    return "qwen/qwen-2.5-vl-7b-instruct" if (image_query or has_image_chunks) else "qwen/qwen-2.5-7b-instruct"


def build_rag_prompt(query: str, context: str, requested_points: int | None = None) -> str:
    """
    Build a RAG prompt with retrieved context.
    
    Args:
        query: User's question
        context: Retrieved document context
    
    Returns:
        Formatted prompt for LLM
    """
    strict_list_rules = ""
    if requested_points:
        strict_list_rules = f"""

LIST FORMAT REQUIREMENTS (MANDATORY):
- Generate EXACTLY {requested_points} numbered points.
- Use numbering from 1 to {requested_points} with no missing numbers.
- Each point must be complete and contain 1-3 full sentences.
- Never stop early. Do not output fewer than {requested_points} points.
- Use this exact structure for each item:
  {"{index}"}. **Title**
  Complete explanation sentence.
- Do not merge points and do not cut sentences.
- Every point must end with proper punctuation.
- Cover different relevant parts of context; avoid repetition.
- If context is limited, still provide exactly {requested_points} points and mark the affected points with: "Based on limited available context.""".rstrip()

    prompt = f"""You are a grounded document analysis assistant.

Use ONLY the provided document excerpts. Do NOT invent facts.

Rules:
- Preserve attribution using [Document: ...] markers.
- If OCR text is noisy/handwritten, provide a best-effort interpretation and explicitly mention uncertainty.
- Prefer quoting extracted phrases from context when possible.
- Never fabricate names/terms that do not appear in the excerpts.
{strict_list_rules}

Document excerpts:
{context}

User query:
{query}

Provide a concise, evidence-grounded answer with clear source attribution."""
    
    return prompt


async def call_llm_api(
    prompt: str,
    model: str | None = None,
    max_tokens: int = 600,
    temperature: float = 0.2,
) -> str:
    """
    Call OpenRouter Chat Completions API.
    
    Args:
        prompt: Prompt to send to LLM
    
    Returns:
        LLM response text
    
    Raises:
        Exception: If API call fails
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model_name = model or os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct")

    if not api_key:
        raise Exception("OPENROUTER_API_KEY not set")

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Multi-Document RAG"
            }

            logger.info(f"Calling OpenRouter API with model {model_name}...")
            logger.debug(f"Prompt length: {len(prompt)} chars")

            response = await client.post(OPENROUTER_API_URL, headers=headers, json=payload)

            # Check response status
            if response.status_code != 200:
                error_detail = response.text[:500]  # First 500 chars of error
                error_msg = f"OpenRouter API returned status {response.status_code}: {error_detail}"
                logger.error(error_msg)
                raise Exception(error_msg)

            # Parse response
            result = response.json()
            if "choices" not in result or not result["choices"]:
                raise Exception(f"Invalid OpenRouter response: {result}")

            answer = result["choices"][0].get("message", {}).get("content", "")
            answer = (answer or "").strip()

            if not answer:
                error_msg = f"OpenRouter API returned empty response. Full response: {result}"
                logger.error(error_msg)
                raise Exception(error_msg)

            logger.info(f"Received response from LLM ({len(answer)} chars)")
            return answer

    except httpx.ConnectError as e:
        error_msg = f"Cannot connect to OpenRouter API: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except httpx.TimeoutException as e:
        error_msg = f"OpenRouter API timeout after {TIMEOUT}s: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except httpx.HTTPError as e:
        error_msg = f"OpenRouter API HTTP error: {type(e).__name__} - {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error calling LLM: {type(e).__name__} - {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


async def stream_llm_api(
    prompt: str,
    model: str | None = None,
    max_tokens: int = 300,
    temperature: float = 0.2,
) -> AsyncIterator[str]:
    """
    Stream token deltas from OpenRouter Chat Completions API.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model_name = model or os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct")

    if not api_key:
        raise Exception("OPENROUTER_API_KEY not set")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Multi-Document RAG",
    }

    logger.info(f"Streaming OpenRouter response with model {model_name}...")

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream("POST", OPENROUTER_API_URL, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_detail = await response.aread()
                    detail_text = (error_detail.decode("utf-8", errors="ignore") if error_detail else "")[:500]
                    raise Exception(f"OpenRouter streaming API returned status {response.status_code}: {detail_text}")

                async for raw_line in response.aiter_lines():
                    line = (raw_line or "").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue

                    choices = obj.get("choices") or []
                    if not choices:
                        continue

                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")

                    if isinstance(content, str) and content:
                        yield content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                txt = item.get("text") or item.get("content") or ""
                                if txt:
                                    yield str(txt)

    except httpx.ConnectError as e:
        error_msg = f"Cannot connect to OpenRouter API: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except httpx.TimeoutException as e:
        error_msg = f"OpenRouter API timeout after {TIMEOUT}s: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except httpx.HTTPError as e:
        error_msg = f"OpenRouter API HTTP error: {type(e).__name__} - {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error calling LLM: {type(e).__name__} - {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


async def generate_answer(query: str, retrieved_chunks: List[Dict]) -> Dict:
    """
    Generate answer using retrieved chunks and LLM.
    
    Args:
        query: User's question
        retrieved_chunks: Retrieved document chunks
    
    Returns:
        Dictionary with answer and sources
    """
    try:
        # Import here to avoid circular dependency
        from .retriever import format_retrieved_context, extract_sources
        
        # Format context
        context = format_retrieved_context(retrieved_chunks)
        
        requested_points = _extract_requested_point_count(query)
        model_name = _select_model_for_query(query, retrieved_chunks)

        # Build prompt
        prompt = build_rag_prompt(query, context, requested_points=requested_points)

        # Anti-truncation defaults for long/structured outputs
        max_tokens = 750 if requested_points else 600
        temperature = 0.2

        # Call LLM
        answer = await call_llm_api(
            prompt,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Post-generation validation for strict numbered-list requests.
        if requested_points:
            valid, reason = _validate_numbered_list(answer, requested_points)

            # First try continuation path when answer stopped early.
            if not valid and reason.startswith("count_mismatch"):
                try:
                    answer = await _continue_missing_points(
                        query=query,
                        context=context,
                        partial_answer=answer,
                        requested_points=requested_points,
                        model_name=model_name,
                    )
                    valid, reason = _validate_numbered_list(answer, requested_points)
                except Exception as e:
                    logger.warning(f"Continuation for missing points failed: {str(e)}")

            retries = 0
            while not valid and retries < 2:
                logger.warning(f"List validation failed ({reason}); regenerating (attempt {retries + 2}/3)")
                repair_prompt = f"""
The previous answer did not satisfy mandatory list constraints ({reason}).

You MUST regenerate the full final answer now.

MANDATORY CONSTRAINTS:
- EXACTLY {requested_points} numbered points (1 to {requested_points}).
- Each point must be complete with 1-3 full sentences.
- No abrupt endings. Every point ends with punctuation.
- No missing items, no merged points, no partial points.
- Use diverse evidence from context and avoid repeating same idea.
- If context is limited, still produce exactly {requested_points} points and add: "Based on limited available context." where needed.

User query:
{query}

Document excerpts:
{context}
""".strip()
                answer = await call_llm_api(
                    repair_prompt,
                    model=model_name,
                    max_tokens=780,
                    temperature=0.2,
                )
                valid, reason = _validate_numbered_list(answer, requested_points)
                retries += 1
        
        # Extract sources
        sources = extract_sources(retrieved_chunks)
        
        return {
            "answer": answer.strip(),
            "sources": sources
        }
    
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")

        # Safe fallback when LLM is unavailable
        from .retriever import extract_sources

        fallback_text = "LLM service unavailable. Please try again in a moment."
        if retrieved_chunks:
            joined = "\n".join(
                (chunk.get("text", "") or "").strip()
                for chunk in retrieved_chunks[:1]
            ).strip()
            if joined:
                fallback_text = (
                    "LLM service unavailable. Best extracted context:\n\n"
                    f"{joined[:700]}"
                )

        return {
            "answer": fallback_text,
            "sources": extract_sources(retrieved_chunks)
        }


async def generate_answer_stream(query: str, retrieved_chunks: List[Dict]) -> AsyncIterator[str]:
    """
    Stream answer text progressively for UI token-by-token rendering.
    """
    from .retriever import format_retrieved_context

    context = format_retrieved_context(retrieved_chunks)
    requested_points = _extract_requested_point_count(query)
    prompt = build_rag_prompt(query, context, requested_points=requested_points)
    model_name = _select_model_for_query(query, retrieved_chunks)

    # For strict list requests, generate validated full answer first, then stream safely.
    if requested_points:
        answer = await call_llm_api(
            prompt,
            model=model_name,
            max_tokens=780,
            temperature=0.2,
        )

        valid, reason = _validate_numbered_list(answer, requested_points)

        if not valid and reason.startswith("count_mismatch"):
            try:
                answer = await _continue_missing_points(
                    query=query,
                    context=context,
                    partial_answer=answer,
                    requested_points=requested_points,
                    model_name=model_name,
                )
                valid, reason = _validate_numbered_list(answer, requested_points)
            except Exception as e:
                logger.warning(f"Streaming continuation for missing points failed: {str(e)}")

        retries = 0
        while not valid and retries < 2:
            repair_prompt = f"""
Your previous response violated list constraints ({reason}).
Regenerate complete final output with EXACTLY {requested_points} numbered points.
Each point must be complete (1-3 sentences), punctuated, and not truncated.

User query:
{query}

Document excerpts:
{context}
""".strip()
            answer = await call_llm_api(
                repair_prompt,
                model=model_name,
                max_tokens=780,
                temperature=0.2,
            )
            valid, reason = _validate_numbered_list(answer, requested_points)
            retries += 1

        # Emit by line blocks to preserve numbered item structure.
        lines = answer.splitlines(keepends=True)
        for line in lines:
            if line:
                yield line
        if not lines:
            yield answer
        return

    async for token in stream_llm_api(
        prompt,
        model=model_name,
        max_tokens=700,
        temperature=0.2,
    ):
        yield token
