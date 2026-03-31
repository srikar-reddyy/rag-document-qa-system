"""
Chat endpoint for RAG-powered question answering.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from schemas.chat import ChatRequest, ChatResponse
from rag.pipeline import rag_query, classify_query, run_count_pipeline
from rag.vectordb import get_all_document_metadata
from rag.vectordb import get_document_chunks
from rag.retriever import retrieve, extract_sources
from rag.generator import generate_answer_stream, generate_answer
from rag.pipeline import build_dynamic_sources, extract_highlights_from_sources
from services import get_chat_service
import logging
import re
import asyncio
import json

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def _extract_requested_point_count(query: str) -> int | None:
    q = (query or "").lower()
    patterns = [
        r"\b(?:top|list|give|provide|share|tell)\s+(\d{1,2})\b",
        r"\b(\d{1,2})\s+(?:key\s+)?(?:points?|items?|reasons?|insights?|takeaways?|facts?|differences?)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 20:
                    return n
            except Exception:
                continue
    return None


def _looks_like_incomplete_list_tail(text: str) -> bool:
    """
    Detect if the last visible line is a likely incomplete list item.
    """
    if not text:
        return False

    tail = text.split("\n")[-1].strip()
    if not tail:
        return False

    if not re.match(r"^(?:[-*]|\d+\.)\s+", tail):
        return False

    # If list tail has no terminating punctuation and no newline yet, keep buffering.
    return not re.search(r"[.!?]\s*$", tail)


def _find_flush_boundary(buffer: str) -> int:
    """
    Find safe flush boundary index (exclusive): newline or sentence end.
    Returns -1 when no safe boundary exists yet.
    """
    if not buffer:
        return -1

    last_newline = buffer.rfind("\n")
    sentence_end = -1
    for m in re.finditer(r"[.!?](?=\s|$)", buffer):
        sentence_end = m.end()

    boundary = max(last_newline + 1 if last_newline >= 0 else -1, sentence_end)
    if boundary <= 0:
        return -1

    prefix = buffer[:boundary]
    if _looks_like_incomplete_list_tail(prefix):
        # If the flush candidate still ends with incomplete list item, back off to previous newline.
        prev_newline = prefix[:-1].rfind("\n")
        if prev_newline >= 0:
            return prev_newline + 1
        return -1

    return boundary


@router.get("/history")
async def get_chat_history():
    """
    Get chat history.
    
    Returns:
        List of chat messages
    """
    chat_service = get_chat_service()
    messages = chat_service.get_history()
    
    return {
        "messages": messages,
        "count": len(messages)
    }


@router.delete("/history")
async def clear_chat_history():
    """
    Clear chat history.
    
    Returns:
        Success message
    """
    chat_service = get_chat_service()
    chat_service.clear_history()
    
    return {
        "message": "Chat history cleared successfully"
    }


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process chat message using RAG (Retrieval-Augmented Generation).
    
    RAG Pipeline:
        1. Classify query type (count, compare, summary, qa)
        2. If count query: use deterministic word counter (no LLM)
        3. Otherwise: Retrieve relevant document chunks from ChromaDB
        4. Construct prompt with retrieved context
        5. Generate LLM response grounded in document context
        6. Return answer with source citations
        7. Save to chat history
    
    Args:
        request: ChatRequest with user message and optional conversation history
    
    Returns:
        ChatResponse with assistant's reply (grounded in uploaded documents)
    
    Raises:
        HTTPException: If RAG query fails
    """
    chat_service = get_chat_service()
    
    logger.info(f"Chat message: {request.message[:100]}...")
    logger.info(f"Selected documents: {request.selected_documents}")
    
    # Classify query type
    query_type = classify_query(request.message)
    logger.info(f"Query type detected: {query_type}")
    
    # Save user message to history
    chat_service.add_message("user", request.message)
    
    try:
        result = None
        
        # Handle count queries without requiring selected documents
        # (they extract doc name from query text itself)
        if query_type == "count":
            logger.info("Count query detected - using deterministic word counter (no LLM)")
            
            # Get list of all available documents
            metadata_list = get_all_document_metadata()
            available_docs = [m["file_name"] for m in metadata_list]
            
            if not available_docs:
                raise HTTPException(
                    status_code=400,
                    detail="No documents available to count words in"
                )
            
            # Run word count pipeline (NO LLM CALL)
            result = run_count_pipeline(request.message, available_docs)
        else:
            # For all other queries (QA, compare, summary), require selected documents
            if not request.selected_documents or len(request.selected_documents) == 0:
                logger.warning("No documents selected for query")
                raise HTTPException(
                    status_code=400, 
                    detail="Please select at least one document to query"
                )
            
            # Execute RAG pipeline with document filtering
            requested_points = _extract_requested_point_count(request.message)
            adaptive_top_k = max(5, min(14, (requested_points + 4) if requested_points else 5))
            result = await rag_query(
                request.message, 
                top_k=adaptive_top_k,
                selected_document_ids=request.selected_documents
            )
        
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error in RAG pipeline")
            logger.error(f"Query failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Extract answer, sources, and highlights
        answer = result.get("answer", "No answer generated.")
        sources = result.get("sources", [])
        highlights = result.get("highlights", [])
        
        # Backward-compatible response text (answer only)
        response_text = answer
        
        # Save assistant response to history
        chat_service.add_message("assistant", response_text, sources, highlights)
        
        logger.info(f"RAG response generated with {len(sources)} sources and {len(highlights)} highlights")
        return ChatResponse(
            response=response_text,
            answer=answer,
            sources=sources,
            highlights=highlights,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error in RAG query: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat responses progressively for real-time UX.
    """
    chat_service = get_chat_service()
    logger.info(f"Streaming chat message: {request.message[:100]}...")
    logger.info(f"Selected documents: {request.selected_documents}")

    query_type = classify_query(request.message)
    chat_service.add_message("user", request.message)

    async def stream_generator():
        answer_parts = []
        sources = []
        highlights = []
        stream_buffer = ""
        retrieved_chunks = []

        def _event(token: str = "", done: bool = False, ev_sources=None, ev_highlights=None, error: str = "") -> str:
            payload = {
                "token": token,
                "done": done,
                "sources": ev_sources if ev_sources is not None else [],
                "highlights": ev_highlights if ev_highlights is not None else [],
            }
            if error:
                payload["error"] = error
            return json.dumps(payload, ensure_ascii=False) + "\n"

        def _is_direct_image_selection(doc_ids: list[str]) -> bool:
            if not doc_ids or len(doc_ids) != 1:
                return False
            try:
                chunks = get_document_chunks(doc_ids, max_chunks_per_document=1)
                if not chunks:
                    return False
                metadata = chunks[0].get("metadata", {}) or {}
                return (metadata.get("file_type") or "").lower() == "image"
            except Exception:
                return False

        async def buffered_emit(text: str):
            nonlocal stream_buffer
            stream_buffer += text

            while True:
                boundary = _find_flush_boundary(stream_buffer)
                if boundary <= 0:
                    break

                out = stream_buffer[:boundary]
                stream_buffer = stream_buffer[boundary:]
                if out:
                    answer_parts.append(out)
                    yield _event(token=out, done=False)
                    await asyncio.sleep(0.015)

        async def flush_remaining():
            nonlocal stream_buffer
            if stream_buffer:
                answer_parts.append(stream_buffer)
                yield _event(token=stream_buffer, done=False)
                stream_buffer = ""

        try:
            # Count queries: deterministic response, streamed character-by-character
            if query_type == "count":
                metadata_list = get_all_document_metadata()
                available_docs = [m["file_name"] for m in metadata_list]
                if not available_docs:
                    text = "No documents available to count words in"
                    async for out in buffered_emit(text):
                        yield out
                    async for out in flush_remaining():
                        yield out
                    yield _event(done=True, ev_sources=[], ev_highlights=[])
                    return

                result = run_count_pipeline(request.message, available_docs)
                text = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                async for out in buffered_emit(text):
                    yield out
                async for out in flush_remaining():
                    yield out
                yield _event(done=True, ev_sources=sources, ev_highlights=[])
                return

            # Non-count queries require document selection
            if not request.selected_documents:
                text = "Please select at least one document to query."
                async for out in buffered_emit(text):
                    yield out
                async for out in flush_remaining():
                    yield out
                yield _event(done=True, ev_sources=[], ev_highlights=[])
                return

            # Retrieval (summary gets broad document chunks, others use semantic retrieval)
            if _is_direct_image_selection(request.selected_documents):
                logger.info("Streaming direct image QA path activated")
                retrieved_chunks = get_document_chunks(
                    request.selected_documents,
                    max_chunks_per_document=3,
                )
            elif query_type == "summary":
                retrieved_chunks = get_document_chunks(
                    request.selected_documents,
                    max_chunks_per_document=25,
                )
            else:
                requested_points = _extract_requested_point_count(request.message)
                adaptive_top_k = max(4, min(14, (requested_points + 3) if requested_points else 4))
                adaptive_broad_k = max(15, adaptive_top_k * 4)
                retrieved_chunks = retrieve(
                    request.message,
                    top_k=adaptive_top_k,
                    broad_k=adaptive_broad_k,
                    selected_document_ids=request.selected_documents,
                )

            if not retrieved_chunks:
                text = "No relevant information found in uploaded documents."
                async for out in buffered_emit(text):
                    yield out
                async for out in flush_remaining():
                    yield out
                yield _event(done=True, ev_sources=[], ev_highlights=[])
                return

            sources = extract_sources(retrieved_chunks)
            logger.info(f"Stream retrieval stats | chunks={len(retrieved_chunks)} | initial_sources={len(sources)}")

            try:
                async for token in generate_answer_stream(request.message, retrieved_chunks):
                    async for out in buffered_emit(token):
                        yield out
            except Exception as stream_model_error:
                logger.warning(f"Primary streaming model path failed: {str(stream_model_error)}")
                try:
                    fallback = await generate_answer(request.message, retrieved_chunks)
                    fallback_text = (fallback.get("answer") or "").strip()
                    if not fallback_text:
                        fallback_text = "I could not stream the model response, but I extracted supporting context."
                    async for out in buffered_emit(fallback_text):
                        yield out
                except Exception as fallback_error:
                    logger.error(f"Fallback non-stream generation failed: {str(fallback_error)}")
                    # Final safety: emit concise context-backed response
                    first_span = ((retrieved_chunks[0].get("text") if retrieved_chunks else "") or "").strip()
                    safety_text = (
                        "I couldn't complete generation from the model. "
                        "Best extracted context:\n\n"
                        f"{first_span[:700]}"
                    ).strip()
                    async for out in buffered_emit(safety_text):
                        yield out

            async for out in flush_remaining():
                yield out

            final_answer = "".join(answer_parts).strip() or "No answer generated."
            try:
                dynamic_sources = await build_dynamic_sources(request.message, final_answer, retrieved_chunks, max_sources=5)
                sources = dynamic_sources or sources
                highlights = extract_highlights_from_sources(sources)
            except Exception as source_error:
                logger.warning(f"Streaming post-source attribution failed: {str(source_error)}")
                highlights = []

            yield _event(done=True, ev_sources=sources, ev_highlights=highlights)

        except Exception as e:
            logger.error(f"Streaming query failed: {str(e)}")
            error_text = "\n\n[Error: unable to stream response]"
            async for out in buffered_emit(error_text):
                yield out
            async for out in flush_remaining():
                yield out
            yield _event(done=True, ev_sources=sources, ev_highlights=highlights, error=str(e))
        finally:
            final_answer = "".join(answer_parts).strip() or "No answer generated."
            chat_service.add_message("assistant", final_answer, sources, highlights)

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
