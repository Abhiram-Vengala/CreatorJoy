import os
import json
from groq import Groq
from typing import Generator
from retrieval import retrieve_relevant_chunks, build_context_block
from models import ChatMessage, RetrievedChunk

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are CreatorJoy AI, an expert content analyst helping creators understand why their videos perform the way they do.

You have access to transcripts from two videos the creator has submitted for comparison. Your job is to:
1. Answer questions about WHY one video outperformed another
2. Analyze hooks (first 5-10 seconds), pacing, language, structure
3. Suggest specific, actionable improvements
4. Cite the exact part of the transcript you're referencing (e.g. "In Video A at ~1:23, you said '...' which...")

Always be specific. Never give generic advice like "improve your hook." Instead say WHAT to change and WHY based on the actual transcript content.

When citing sources, reference the video title and approximate timestamp.
Format your response in clear paragraphs. Use bullet points only for actionable suggestions.
"""


def build_messages(
    user_message: str,
    history: list[ChatMessage],
    context_block: str,
) -> list[dict]:
    """
    Build the full messages array for the Groq API call.

    Memory strategy: pass full history each turn (simple, no vector store needed).
    This works fine up to ~10 turns before context gets long.
    For production: summarize history > 10 turns with a cheap model call.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject RAG context as a system-level reference block
    if context_block:
        messages.append(
            {
                "role": "system",
                "content": f"Here are the relevant transcript excerpts for this question:\n\n{context_block}",
            }
        )

    # Add conversation history
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    return messages


def stream_chat_response(
    user_message: str,
    session_id: str,
    history: list[ChatMessage],
) -> Generator[str, None, None]:
    """
    Full RAG + streaming pipeline:
    1. Retrieve relevant chunks from Pinecone
    2. Build context block
    3. Stream Groq response token by token
    4. Yield SSE-formatted chunks for FastAPI StreamingResponse

    Yields strings in SSE format:
      data: {"type": "token", "content": "..."}\n\n
      data: {"type": "sources", "sources": [...]}\n\n
      data: {"type": "done"}\n\n
    """
    # Step 1: Retrieve
    chunks: list[RetrievedChunk] = retrieve_relevant_chunks(
        query=user_message,
        session_id=session_id,
        top_k=6,
    )

    # Step 2: Build context
    context_block = build_context_block(chunks)

    # Step 3: Build messages
    messages = build_messages(user_message, history, context_block)

    # Step 4: Stream from Groq
    # llama-3.3-70b-versatile: best free model on Groq for reasoning tasks
    stream = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        stream=True,
    )

    # Yield sources — child_text shown as citation snippet in UI
    # parent_text is what the LLM actually used (larger context)
    sources_payload = [
        {
            "chunk_id": c.chunk_id,
            "video_id": c.video_id,
            "video_title": c.video_title,
            "text": c.child_text[:200] + "..." if len(c.child_text) > 200 else c.child_text,
            "timestamp_approx": c.timestamp_approx,
            "score": c.score,
        }
        for c in chunks
    ]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload})}\n\n"

    # Stream tokens
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            payload = json.dumps({"type": "token", "content": delta.content})
            yield f"data: {payload}\n\n"

    # Signal completion
    yield f"data: {json.dumps({'type': 'done'})}\n\n"