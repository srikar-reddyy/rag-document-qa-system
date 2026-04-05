"""
Pydantic schemas for chat-related requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional


class HighlightItem(BaseModel):
    """Grounding span used by viewer highlighting."""
    doc_name: str = Field(..., description="Source document name")
    page: int = Field(default=1, description="1-based page number")
    text: str = Field(..., description="Exact evidence span")
    score: float = Field(default=0.0, description="Similarity/confidence score")
    char_start: Optional[int] = Field(default=None, description="Optional start char offset")
    char_end: Optional[int] = Field(default=None, description="Optional end char offset")
    boxes: list[list[float]] = Field(default_factory=list, description="Word-level [x1, y1, x2, y2] boxes")
    bbox: Optional[list[float]] = Field(default=None, description="Optional backward-compatible primary box")


class SourceItem(BaseModel):
    """Simple source citation item."""
    doc: Optional[str] = Field(default=None, description="Source document name")
    file: str = Field(..., description="Source file name")
    page: int = Field(default=1, description="1-based page number")
    text: Optional[str] = Field(default=None, description="Supporting evidence sentence")
    score: Optional[float] = Field(default=None, description="Evidence confidence/relevance score")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, description="User's question or message")
    selected_documents: list[str] = Field(default=[], description="List of document IDs to query")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Backward-compatible combined reply")
    answer: str = Field(..., description="LLM-generated answer only")
    sources: list[SourceItem] = Field(default_factory=list, description="Source citations")
    highlights: list[HighlightItem] = Field(default_factory=list, description="Viewer highlight spans")


class UploadResponse(BaseModel):
    """Response model for document upload endpoint."""
    message: str = Field(..., description="Upload status message")
    files: list[str] = Field(..., description="List of uploaded file names")
