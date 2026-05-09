import os
from dotenv import load_dotenv
from pinecone import Pinecone
from embeddings import embed_query, INDEX_NAME
from models import RetrievedChunk

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
_pc = Pinecone(api_key=PINECONE_API_KEY)


def retrieve_relevant_chunks(
    query: str,
    session_id: str,
    top_k: int = 6,
) -> list[RetrievedChunk]:
    """
    Embed the user query and retrieve top_k most relevant child chunks from Pinecone.

    Why top_k=6?
    - 6 child matches → typically 2-4 unique parents (siblings often match together)
    - We deduplicate by parent_id so LLM doesn't get the same parent context twice
    - After dedup: ~3 unique parents × ~270 tokens = ~810 tokens of context
    - Leaves plenty of room in Groq's 8k context for conversation history + answer

    The returned RetrievedChunk carries both:
    - child_text → shown as citation snippet in the UI (small, precise)
    - parent_text → sent to LLM as context (large, rich)
    """
    index = _pc.Index(INDEX_NAME)
    query_vector = embed_query(query)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=session_id,
        include_metadata=True,
    )

    # Deduplicate by parent_id — siblings from the same parent carry identical
    # parent_text, so sending duplicates just wastes LLM context window tokens
    seen_parents: set[str] = set()
    chunks: list[RetrievedChunk] = []

    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        parent_id = meta.get("parent_id", match["id"])

        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        chunks.append(
            RetrievedChunk(
                chunk_id=match["id"],
                video_id=meta.get("video_id", ""),
                video_title=meta.get("video_title", "Unknown"),
                child_text=meta.get("child_text", ""),       # shown in UI as citation
                parent_text=meta.get("parent_text", ""),     # sent to LLM as context
                timestamp_approx=meta.get("timestamp_approx", 0.0),
                score=round(match.get("score", 0.0), 4),
            )
        )

    return chunks


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved parent contexts into a context string for the LLM prompt.
    Groups by video so the model can clearly distinguish Video A vs Video B.
    Uses parent_text (large) — not child_text — for rich LLM context.
    """
    if not chunks:
        return "No relevant transcript content found."

    by_video: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        by_video.setdefault(chunk.video_title, []).append(chunk)

    context_parts = []
    for video_title, video_chunks in by_video.items():
        context_parts.append(f"=== {video_title} ===")
        for chunk in sorted(video_chunks, key=lambda c: c.timestamp_approx):
            timestamp = _format_timestamp(chunk.timestamp_approx)
            context_parts.append(f"[~{timestamp}] {chunk.parent_text}")
        context_parts.append("")

    return "\n".join(context_parts)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS string."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"