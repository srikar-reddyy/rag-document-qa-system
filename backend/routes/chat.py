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
from rag.generator import generate_answer_stream
from services import get_chat_service
import logging
import re
import asyncio

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
                    yield out
                    await asyncio.sleep(0.015)

        async def flush_remaining():
            nonlocal stream_buffer
            if stream_buffer:
                answer_parts.append(stream_buffer)
                yield stream_buffer
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
                    return

                result = run_count_pipeline(request.message, available_docs)
                text = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                async for out in buffered_emit(text):
                    yield out
                async for out in flush_remaining():
                    yield out
                return

            # Non-count queries require document selection
            if not request.selected_documents:
                text = "Please select at least one document to query."
                async for out in buffered_emit(text):
                    yield out
                async for out in flush_remaining():
                    yield out
                return

            # Retrieval (summary gets broad document chunks, others use semantic retrieval)
            if query_type == "summary":
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
                return

            sources = extract_sources(retrieved_chunks)

            async for token in generate_answer_stream(request.message, retrieved_chunks):
                async for out in buffered_emit(token):
                    yield out

            async for out in flush_remaining():
                yield out

        except Exception as e:
            logger.error(f"Streaming query failed: {str(e)}")
            error_text = "\n\n[Error: unable to stream response]"
            async for out in buffered_emit(error_text):
                yield out
            async for out in flush_remaining():
                yield out
        finally:
            final_answer = "".join(answer_parts).strip() or "No answer generated."
            chat_service.add_message("assistant", final_answer, sources, highlights)

    return StreamingResponse(stream_generator(), media_type="text/plain")
