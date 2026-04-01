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
import base64
import mimetypes
import importlib
import requests

_pdf2image = importlib.util.find_spec("pdf2image")
if _pdf2image is not None:
    convert_from_path = importlib.import_module("pdf2image").convert_from_path
else:
    convert_from_path = None

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
    image_data_urls: List[str] | None = None,
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

    continuation_messages = _build_llm_messages(
        continuation_prompt,
        image_data_urls=image_data_urls or [],
        force_image_instructions=bool(image_data_urls),
    )

    continuation = await call_llm_api(
        continuation_prompt,
        model=model_name,
        max_tokens=780,
        temperature=0.2,
        messages=continuation_messages,
    )

    merged = f"{partial_answer.strip()}\n\n{continuation.strip()}".strip()
    return merged


def _select_model_for_query(query: str, retrieved_chunks: List[Dict]) -> str:
    return "qwen/qwen-2.5-7b-instruct"


def has_image_context(chunks: List[Dict]) -> bool:
    return len(_collect_chunk_image_paths(chunks, max_images=1)) > 0


def is_visual_query(query: str) -> bool:
    query = (query or "").lower()

    visual_keywords = [
        "image", "figure", "diagram", "chart",
        "graph", "plot", "visual", "illustration",
        "screenshot", "picture"
    ]

    return any(word in query for word in visual_keywords)


def route_model(query: str, chunks: List[Dict]) -> str:
    has_visual_context = has_image_context(chunks)
    if is_visual_query(query) and has_visual_context:
        return "vision"
    return "text"


def _collect_chunk_image_paths(chunks: List[Dict], max_images: int = 2) -> List[str]:
    image_paths: List[str] = []
    seen = set()

    for chunk in (chunks or []):
        metadata = chunk.get("metadata", {}) or {}

        raw_paths = metadata.get("image_paths", "")
        for path in str(raw_paths).split("|"):
            cleaned = path.strip()
            if not cleaned or cleaned in seen:
                continue
            image_paths.append(cleaned)
            seen.add(cleaned)
            if len(image_paths) >= max_images:
                return image_paths

        image_file_path = metadata.get("source_image_path") or metadata.get("file_path")
        if image_file_path:
            cleaned = str(image_file_path).strip()
            if cleaned and cleaned not in seen:
                image_paths.append(cleaned)
                seen.add(cleaned)
                if len(image_paths) >= max_images:
                    return image_paths

    return image_paths


def _get_primary_image_path(chunks: List[Dict]) -> str | None:
    paths = _collect_chunk_image_paths(chunks, max_images=1)
    return paths[0] if paths else None


def _build_vision_prompt(query: str, context: str | None = None) -> str:
    base = f"""
You are analyzing an image.

Answer the user's question based ONLY on the image.

User Question:
{query}

If the question asks:
- about color → mention colors
- about count → count carefully
- about meaning → explain concept
- about labels → list exact labels

Be precise. Do not give generic explanations.
""".strip()

    if context and context.strip():
        return f"{base}\n\nContext (use only if it matches the image):\n{context}".strip()
    return base


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vision(prompt, image_paths):
    images_b64 = [encode_image(p) for p in image_paths[:2]]

    resp = requests.post(
        "http://192.168.0.100:11434/api/generate",
        json={
            "model": "qwen2.5vl:3b",
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
        },
        timeout=60,
    )

    data = resp.json()
    return data.get("response", "")


def call_local_vision(image_path: str, query: str) -> str:
    import os

    if not image_path or not os.path.exists(image_path):
        raise Exception("Invalid image path")

    try:
        result = (call_vision(query, [image_path]) or "").strip()
        if not result:
            raise Exception("Empty response from Ollama")

        return result

    except Exception as e:
        raise Exception(f"Ollama request failed: {str(e)}")


def _build_image_analysis_instructions() -> str:
    return ""


def stream_local_vision(image_path: str, query: str):
    import base64
    import requests
    import os
    import json

    if not image_path or not os.path.exists(image_path):
        yield "Image not available."
        return

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = requests.post(
        "http://192.168.0.100:11434/api/generate",
        json={
            "model": "qwen2.5vl:3b",
            "prompt": query,
            "images": [img_b64],
            "stream": True,
        },
        stream=True,
        timeout=120,
    )

    if response.status_code != 200:
        raise Exception(f"Ollama streaming error: {response.text[:400]}")

    for line in response.iter_lines():
        if not line:
            continue
        try:
            data = json.loads(line.decode("utf-8"))
            if data.get("done") is True:
                break
            if "response" in data and data["response"]:
                yield data["response"]
        except Exception:
            continue


def _image_file_to_data_url(image_path: str) -> str | None:
    try:
        if not image_path or not os.path.exists(image_path):
            return None
        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{image_b64}"
    except Exception as e:
        logger.warning(f"Failed to encode image path {image_path}: {str(e)}")
        return None


def _pdf_page_to_data_url(pdf_path: str, page_number: int) -> str | None:
    if convert_from_path is None:
        return None

    try:
        if not pdf_path or not os.path.exists(pdf_path):
            return None

        images = convert_from_path(
            pdf_path,
            first_page=max(1, int(page_number)),
            last_page=max(1, int(page_number)),
            dpi=200,
            thread_count=1,
        )
        if not images:
            return None

        import io
        buffer = io.BytesIO()
        images[0].save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{image_b64}"
    except Exception as e:
        logger.warning(f"Failed to encode PDF page image {pdf_path}#p{page_number}: {str(e)}")
        return None


def _collect_image_data_urls(retrieved_chunks: List[Dict], max_images: int = 2) -> List[str]:
    urls: List[str] = []
    seen = set()

    for chunk in (retrieved_chunks or []):
        metadata = chunk.get("metadata", {}) or {}
        file_type = (metadata.get("file_type") or "").lower()

        data_url = None
        if file_type == "image":
            image_path = metadata.get("source_image_path")
            data_url = _image_file_to_data_url(image_path)
        elif file_type == "pdf" and metadata.get("has_visual_context"):
            pdf_path = metadata.get("source_pdf_path")
            page_number = int(metadata.get("page", metadata.get("page_number", 1)) or 1)
            data_url = _pdf_page_to_data_url(pdf_path, page_number)

        if not data_url:
            continue
        if data_url in seen:
            continue

        urls.append(data_url)
        seen.add(data_url)

        if len(urls) >= max_images:
            break

    logger.info(f"Image payload selection | selected_images={len(urls)}")
    return urls


def _build_llm_messages(
    prompt: str,
    image_data_urls: List[str] | None = None,
    force_image_instructions: bool = False,
) -> List[Dict]:
    urls = image_data_urls or []
    if not urls:
        return [{"role": "user", "content": prompt}]

    text_payload = prompt
    if force_image_instructions:
        text_payload = prompt

    # Keep multimodal payload bounded to reduce provider-side request failures.
    if len(text_payload) > 9000:
        text_payload = text_payload[:9000] + "\n\n[Context truncated for multimodal stability.]"

    content = [{"type": "text", "text": text_payload}]
    for u in urls:
        content.append({"type": "image_url", "image_url": {"url": u}})

    return [{"role": "user", "content": content}]


def _looks_unknown_response(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return True
    patterns = [
        "unknown",
        "cannot determine",
        "can't determine",
        "not clear",
        "unclear from the image",
    ]
    return any(p in a for p in patterns)


def _count_key_elements(answer: str) -> int:
    if not answer:
        return 0

    # Prefer counting explicit bullets in section 2
    section_match = re.search(
        r"(?is)2\.\s*key\s*elements.*?(?=\n\s*3\.|\Z)",
        answer,
    )
    section = section_match.group(0) if section_match else answer

    bullet_count = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+\))\s+\S+", section))
    if bullet_count > 0:
        return bullet_count

    # Fallback: count comma-separated specific items after colon
    comma_items = re.findall(r"(?i)key\s*elements\s*[:\-]\s*(.+)", section)
    if comma_items:
        last_line = comma_items[-1]
        parts = [p.strip() for p in last_line.split(",") if p.strip()]
        return len(parts)

    return 0


def _passes_image_quality_check(answer: str) -> tuple[bool, str]:
    if not answer or not answer.strip():
        return False, "empty_answer"
    if _looks_unknown_response(answer):
        return False, "unknown_like_response"
    return True, "ok"


def _build_image_repair_prompt(query: str, context: str, reason: str) -> str:
    return (
        f"Previous output failed quality check: {reason}.\n\n"
        f"User question: {query}\n\n"
        f"Context (use only if it matches the image):\n{context}\n"
    )


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
    messages: List[Dict] | None = None,
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
                "messages": messages or [
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
            logger.info(f"LLM payload mode | multimodal={bool(messages and isinstance(messages[0].get('content'), list))}")

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
    messages: List[Dict] | None = None,
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
        "messages": messages or [
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
        vision_image_paths = _collect_chunk_image_paths(retrieved_chunks, max_images=2)
        has_visual_context = len(vision_image_paths) > 0
        visual_query = is_visual_query(query)
        image_mode = route_model(query, retrieved_chunks) == "vision"
        print(f"[DEBUG] Query: {query}")
        print(f"[DEBUG] is_visual_query: {visual_query}")
        print(f"[DEBUG] has_visual_context: {has_visual_context}")
        logger.info(f"User query: {query}")
        logger.info(
            "Generation mode | image_mode=%s | model=%s | chunks=%s",
            image_mode,
            model_name,
            len(retrieved_chunks or []),
        )
        logger.info("Using LOCAL Ollama for vision" if image_mode else "Using OpenRouter for text")
        if image_mode:
            logger.info("Using dynamic image reasoning")

        # Build prompt
        prompt = build_rag_prompt(query, context, requested_points=requested_points)
        messages = _build_llm_messages(
            prompt,
            image_data_urls=[],
            force_image_instructions=False,
        )

        # Anti-truncation defaults for long/structured outputs
        max_tokens = 750 if requested_points else 600
        temperature = 0.2

        # Call LLM
        if image_mode:
            print("[ROUTING] Using VISION model")
            logger.info(f"Image paths: {vision_image_paths}")
            try:
                image_paths = [p for p in vision_image_paths if p.strip()]

                if len(image_paths) > 0:
                    print("[DEBUG] Using VISION (Ollama)")
                else:
                    print("[DEBUG] Using TEXT (OpenRouter)")
                    raise Exception("No image paths found in metadata")

                enhanced_query = _build_vision_prompt(query, context=context)

                logger.info("Calling Ollama vision model...")
                answer = call_vision(enhanced_query, image_paths)
                if not answer or not answer.strip():
                    raise Exception("Empty response from Ollama")
                logger.info(f"Ollama response length: {len(answer)}")

            except Exception as vision_error:
                logger.error(f"Ollama vision failed: {str(vision_error)}")
                answer = await call_llm_api(
                    prompt,
                    model="qwen/qwen-2.5-7b-instruct",
                    max_tokens=700,
                    temperature=0.2,
                    messages=messages,
                )
                if not answer or not answer.strip():
                    answer = "Failed to process image. Please try again."
        else:
            print("[ROUTING] Using TEXT model (OpenRouter)")
            answer = await call_llm_api(
                prompt,
                model="qwen/qwen-2.5-7b-instruct",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            print("[DEBUG] Using TEXT (OpenRouter)")

        if image_mode:
            valid_img, img_reason = _passes_image_quality_check(answer)
            if not valid_img:
                logger.warning(f"Image answer failed quality check: {img_reason}")

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
                        image_data_urls=[],
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
                    messages=_build_llm_messages(
                        repair_prompt,
                        image_data_urls=[],
                        force_image_instructions=False,
                    ),
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
        logger.error(f"Critical error: {str(e)}")

        # Safe fallback when LLM is unavailable
        from .retriever import extract_sources

        return {
            "answer": "Failed to process image. Please try again.",
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
    vision_image_paths = _collect_chunk_image_paths(retrieved_chunks, max_images=2)
    has_visual_context = len(vision_image_paths) > 0
    visual_query = is_visual_query(query)
    image_mode = route_model(query, retrieved_chunks) == "vision"
    print(f"[DEBUG] Query: {query}")
    print(f"[DEBUG] is_visual_query: {visual_query}")
    print(f"[DEBUG] has_visual_context: {has_visual_context}")
    messages = _build_llm_messages(
        prompt,
        image_data_urls=[],
        force_image_instructions=False,
    )
    logger.info(f"User query: {query}")
    logger.info(
        "Streaming mode | image_mode=%s | model=%s | chunks=%s",
        image_mode,
        model_name,
        len(retrieved_chunks or []),
    )
    logger.info("Using LOCAL Ollama for vision" if image_mode else "Using OpenRouter for text")

    # For strict list requests, generate validated full answer first, then stream safely.
    if requested_points:
        answer = await call_llm_api(
            prompt,
            model=model_name,
            max_tokens=780,
            temperature=0.2,
            messages=messages,
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
                    image_data_urls=[],
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
                messages=_build_llm_messages(
                    repair_prompt,
                    image_data_urls=[],
                    force_image_instructions=False,
                ),
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

    if image_mode and not requested_points:
        print("[ROUTING] Using VISION model")
        logger.info(f"Image paths: {vision_image_paths}")
        if not vision_image_paths:
            logger.error("No image path found in metadata")
            yield "Image not available."
            return

        print("[DEBUG] Using VISION (Ollama)")
        logger.info("Using dynamic image reasoning")
        enhanced_query = _build_vision_prompt(query, context=context)
        answer = call_vision(enhanced_query, vision_image_paths)
        if answer:
            yield answer
        return

    if not image_mode:
        print("[ROUTING] Using TEXT model (OpenRouter)")

    try:
        async for token in stream_llm_api(
            prompt,
            model=model_name,
            max_tokens=700,
            temperature=0.2,
            messages=messages,
        ):
            yield token
    except Exception as stream_error:
        logger.warning(f"Streaming LLM call failed, falling back to non-stream response: {str(stream_error)}")
        fallback_answer = await call_llm_api(
            prompt,
            model=model_name,
            max_tokens=700,
            temperature=0.2,
            messages=messages,
        )
        if _looks_unknown_response(fallback_answer) and image_mode:
            strict_prompt = _build_vision_prompt(query, context=context)
            strict_messages = _build_llm_messages(
                strict_prompt,
                image_data_urls=[],
                force_image_instructions=False,
            )
            fallback_answer = await call_llm_api(
                strict_prompt,
                model="qwen/qwen-2.5-7b-instruct",
                max_tokens=780,
                temperature=0.15,
                messages=strict_messages,
            )
        yield fallback_answer
