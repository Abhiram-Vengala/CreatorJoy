import uuid
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()

from ingestion import ingest_video
from embeddings import embed_and_store
from chat import stream_chat_response
from models import IngestRequest, IngestResponse, ChatRequest

app = FastAPI(title="CreatorJoy API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest two YouTube video URLs:
    1. Fetch transcripts + metadata for both
    2. Chunk, embed, upsert to Pinecone under a new session_id
    3. Return session_id + video metadata to frontend

    session_id is a UUID generated per request — acts as Pinecone namespace
    so different users' data never collide.
    """
    try:
        # Ingest both videos concurrently
        video_a, video_b = await asyncio.gather(
            ingest_video(request.url_a),
            ingest_video(request.url_b),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Embed and store in Pinecone
    try:
        store_result = embed_and_store(video_a, video_b, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    return IngestResponse(
        session_id=session_id,
        video_a=video_a,
        video_b=video_b,
        message=f"Ingested {store_result['total_vectors']} chunks into Pinecone.",
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using SSE (Server-Sent Events).

    Returns a StreamingResponse that yields:
      - sources: relevant transcript chunks used as context
      - token: individual LLM response tokens
      - done: completion signal

    Frontend consumes this with EventSource or fetch + ReadableStream.
    """
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    def generate():
        yield from stream_chat_response(
            user_message=request.message,
            session_id=request.session_id,
            history=request.history,
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )