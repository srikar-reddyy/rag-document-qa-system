"""
Main FastAPI application for Multi-Document Reasoning Engine.
Phase 1: Chat + Document Upload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import API_TITLE, API_VERSION, API_DESCRIPTION
from routes import chat, upload, debug, compare
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from backend/.env
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(debug.router)
app.include_router(compare.router)


@app.on_event("startup")
async def startup_event():
    """Preload models and initialize services on startup for faster first request."""
    logger.info("Initializing services...")
    
    # Initialize services (loads metadata and chat history)
    from services import get_document_service, get_chat_service
    doc_service = get_document_service()
    chat_service = get_chat_service()
    
    logger.info(f"✓ Loaded {len(doc_service.get_all_documents())} documents")
    logger.info(f"✓ Loaded {len(chat_service.get_history())} chat messages")
    
    # Preload embedding model
    logger.info("Preloading embedding model...")
    try:
        from rag.embedder import get_embedding_model
        model = get_embedding_model()
        logger.info(f"✓ Embedding model loaded: {model}")
    except Exception as e:
        logger.warning(f"Could not preload embedding model: {e}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Multi-Document Reasoning Engine API - Phase 1",
        "endpoints": {
            "chat": "/chat",
            "compare": "/compare",
            "upload": "/upload",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check for monitoring."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
