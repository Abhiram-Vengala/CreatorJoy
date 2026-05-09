from pydantic import BaseModel
from typing import Optional


class TranscriptChunk(BaseModel):
    """Legacy single-level chunk — kept for backwards compat."""
    chunk_id: str
    video_id: str
    video_title: str
    text: str
    chunk_index: int
    timestamp_approx: float


class ChildChunk(BaseModel):
    """
    Small chunk (~60-80 words) — embedded and stored in Pinecone.
    Used for precise retrieval. Its parent_text is what actually
    gets sent to the LLM as context.
    """
    child_id: str            # "{video_id}#parent{p_idx}#child{c_idx}"
    parent_id: str           # "{video_id}#parent{p_idx}"
    video_id: str
    video_title: str
    child_text: str          # small — embedded into Pinecone
    parent_text: str         # large — sent to LLM after retrieval
    child_index: int
    parent_index: int
    timestamp_approx: float


class ParentChunk(BaseModel):
    """
    Large chunk (~200-250 words) — NOT embedded, only stored as context.
    Lives in child metadata so no second DB is needed.
    """
    parent_id: str
    video_id: str
    video_title: str
    text: str
    parent_index: int
    timestamp_approx: float
    children: list[ChildChunk] = []


class VideoData(BaseModel):
    video_id: str
    url: str
    title: str
    author: str
    thumbnail_url: str
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    engagement_rate: Optional[float] = None
    transcript_chunks: list[TranscriptChunk] = []   # legacy fallback
    parent_chunks: list[ParentChunk] = []            # parent-child hierarchy
    raw_transcript: str


class IngestRequest(BaseModel):
    url_a: str
    url_b: str


class IngestResponse(BaseModel):
    session_id: str
    video_a: VideoData
    video_b: VideoData
    message: str


class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: list[ChatMessage] = []


class RetrievedChunk(BaseModel):
    chunk_id: str
    video_id: str
    video_title: str
    child_text: str          # small — shown as citation snippet in UI
    parent_text: str         # large — sent to LLM as actual context
    timestamp_approx: float
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]