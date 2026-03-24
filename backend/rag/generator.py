"""
Answer Generator
Constructs prompts and calls remote Qwen API
"""

import httpx
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Remote LLM API Configuration
LLM_API_URL = "https://umbellar-mechelle-supernationally.ngrok-free.dev/chat"
TIMEOUT = 120.0


def build_rag_prompt(query: str, context: str) -> str:
    """
    Build a RAG prompt with retrieved context.
    
    Args:
        query: User's question
        context: Retrieved document context
    
    Returns:
        Formatted prompt for LLM
    """
    prompt = f"""You are a grounded document analysis assistant.

Use ONLY the provided document excerpts. Do NOT invent facts.

Rules:
- Preserve attribution using [Document: ...] markers.
- If OCR text is noisy/handwritten, provide a best-effort interpretation and explicitly mention uncertainty.
- Prefer quoting extracted phrases from context when possible.
- Never fabricate names/terms that do not appear in the excerpts.

Document excerpts:
{context}

User query:
{query}

Provide a concise, evidence-grounded answer with clear source attribution."""
    
    return prompt


async def call_llm_api(prompt: str) -> str:
    """
    Call the remote Qwen API.
    
    Args:
        prompt: Prompt to send to LLM
    
    Returns:
        LLM response text
    
    Raises:
        Exception: If API call fails
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {"prompt": prompt}
            
            logger.info(f"Calling remote LLM API at {LLM_API_URL}...")
            logger.debug(f"Prompt length: {len(prompt)} chars")
            
            response = await client.post(LLM_API_URL, json=payload)
            
            # Check response status
            if response.status_code != 200:
                error_detail = response.text[:500]  # First 500 chars of error
                error_msg = f"LLM API returned status {response.status_code}: {error_detail}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Parse response
            result = response.json()
            answer = result.get("response", "")
            
            if not answer:
                error_msg = f"LLM API returned empty response. Full response: {result}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            logger.info(f"Received response from LLM ({len(answer)} chars)")
            return answer
    
    except httpx.ConnectError as e:
        error_msg = f"Cannot connect to LLM API at {LLM_API_URL}: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    except httpx.TimeoutException as e:
        error_msg = f"LLM API timeout after {TIMEOUT}s: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    except httpx.HTTPError as e:
        error_msg = f"LLM API HTTP error: {type(e).__name__} - {str(e)}"
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
        
        # Build prompt
        prompt = build_rag_prompt(query, context)
        
        # Call LLM
        answer = await call_llm_api(prompt)
        
        # Extract sources
        sources = extract_sources(retrieved_chunks)
        
        return {
            "answer": answer.strip(),
            "sources": sources
        }
    
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise
