"""
Pydantic schemas for compare mode.
"""

from pydantic import BaseModel, Field


class CompareRequest(BaseModel):
    """Request payload for compare endpoint."""

    doc_ids: list[str] = Field(
        ...,
        min_items=2,
        description="List of selected document IDs (minimum two).",
    )
    query: str = Field(..., min_length=1, description="Comparison query")


class CompareHighlightItem(BaseModel):
    """Source-aligned compare highlight."""

    text: str = Field(..., description="Source-aligned highlight text")
    page: int = Field(default=1, description="1-based page number")


class CompareDocumentResult(BaseModel):
    """Per-document compare output."""

    doc_id: str = Field(..., description="Document ID")
    doc_name: str = Field(..., description="Document file name")
    text: str = Field(..., description="Document-specific answer")
    highlights: list[CompareHighlightItem] = Field(default_factory=list, description="Section/block highlight texts")


class CompareSummaryTopic(BaseModel):
    """Detailed comparison item."""

    topic: str = Field(..., description="Section topic title")
    details: str = Field(..., description="Detailed comparison explanation for the topic")


class CompareSummaryDifferences(BaseModel):
    """Structured differences by document."""

    docA: list[CompareSummaryTopic] = Field(default_factory=list, description="Differences specific to Document A")
    docB: list[CompareSummaryTopic] = Field(default_factory=list, description="Differences specific to Document B")


class CompareSummary(BaseModel):
    """Structured compare summary."""

    overview: str = Field(default="", description="High-level comparison summary")
    similarities: list[CompareSummaryTopic] = Field(default_factory=list, description="Common points across both documents")
    differences: CompareSummaryDifferences = Field(default_factory=CompareSummaryDifferences)


class CompareResponse(BaseModel):
    """Final compare response payload."""

    summary: CompareSummary
    intent: str = Field(..., description="Extracted comparison intent")
    docA: CompareDocumentResult
    docB: CompareDocumentResult