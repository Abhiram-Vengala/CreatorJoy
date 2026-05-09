# CreatorJoy — Video Intelligence Challenge

A full-stack RAG application that takes two YouTube video URLs, pulls their transcripts, and lets creators chat about comparisons, hooks, and improvements — with streaming responses, source citations, and conversation memory.

Built as a technical challenge submission for CreatorJoy.

---

## Demo

> Paste two YouTube URLs → click Analyse → ask anything about why one video crushed it

**Example questions the system handles:**
- "Why did Video A outperform Video B?"
- "Compare the hooks in the first 10 seconds"
- "What specific improvements would you suggest for my opening?"
- "Which video has better audience retention signals?"

---

## Architecture

```
User
 │
 ├── React + Vite (localhost:5173)
 │     ├── VideoInput     → paste 2 URLs, shows thumbnails after ingest
 │     ├── ChatPanel      → streaming chat with SSE, conversation memory
 │     └── SourceCitations → cited transcript chunks with timestamps
 │
 └── FastAPI (localhost:8000)
       ├── POST /ingest   → ingests both videos concurrently
       └── POST /chat/stream → RAG pipeline, streams tokens via SSE
```

### RAG Pipeline

```
YouTube URL
    │
    ▼
extract_video_id()
    │
    ├── fetch_video_metadata()   oEmbed API (no key needed)
    └── fetch_transcript()       youtube-transcript-api
              │
              ▼
    build_parent_child_chunks()
              │
    ┌─────────┴──────────┐
    │  Parent (~200 words) │  ← broad context, NOT embedded
    │  ├── Child 1 (~65w)  │  ← embedded → Pinecone
    │  ├── Child 2 (~65w)  │  ← embedded → Pinecone
    │  └── Child 3 (~65w)  │  ← embedded → Pinecone
    └────────────────────┘
              │
              ▼
    all-MiniLM-L6-v2 embeddings (384-dim, local, free)
              │
              ▼
    Pinecone upsert (namespaced by session_id)
              │
    ── at query time ──────────────────────────────
              │
    embed query → Pinecone top-k → deduplicate by parent_id
              │
    parent_text → LLM context     (rich, ~270 tokens)
    child_text  → UI citation     (precise, ~85 tokens)
              │
              ▼
    Groq llama-3.3-70b-versatile (streaming)
              │
              ▼
    SSE tokens → React EventStream → rendered response
```

---

## Chunking Strategy: Why Parent-Child?

Most RAG systems use flat fixed-size chunking. We use **parent-child hierarchical chunking** instead.

**The problem with flat chunking:**
- Large chunks (300+ words) → embeddings are diluted, poor cosine similarity with short queries
- Small chunks (~50 words) → retrieval is precise but LLM gets too little context to reason well

**Parent-child solves both:**

| | Flat Chunking | Parent-Child |
|---|---|---|
| Embedded unit | 300 words | 65 words (child) |
| LLM context unit | same 300 words | 200 words (parent) |
| Query-chunk size match | poor | good |
| Context richness | moderate | high |
| Vectors at scale | more | fewer (1 parent = 3 children, deduped at retrieval) |

**Key implementation detail:** Parent text is stored directly in each child's Pinecone metadata. No second database (Redis, DynamoDB) needed. When a child is retrieved, we read `parent_text` from its metadata and send that to the LLM — single lookup, zero extra infra.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | React + Vite | Fast dev, clean component model |
| Backend | FastAPI | Async-native, ideal for SSE streaming |
| Transcripts | youtube-transcript-api ≥1.1.0 | No YouTube API key needed |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) | Free, local, 384-dim, fast |
| Vector DB | Pinecone (serverless, free tier) | Zero-ops, namespace isolation per session |
| LLM | Groq llama-3.3-70b-versatile | Free tier, fast inference, strong reasoning |
| Streaming | FastAPI StreamingResponse + SSE | Real-time token delivery to frontend |
| Memory | Full history passed per turn | Simple, no extra infra, works up to ~10 turns |

---

## Cost Analysis at Scale

**1,000 creators/day, avg 10-min video (~1,500 words):**

| Resource | Usage | Cost |
|---|---|---|
| Vectors per session | ~48 (2 videos × 8 parents × 3 children) | — |
| Vectors/day | ~48,000 | Within Pinecone free tier (100k) |
| Groq inference | ~2,000 tokens/query | Free tier: 6k req/min |
| all-MiniLM embeddings | Local CPU | $0 |
| **Total at free tier** | **~1,000 sessions/day** | **~$0** |

**At 10,000 creators/day:** migrate to Qdrant self-hosted (~$50/mo VPS) + Groq paid tier (~$0.06/1M tokens). Pinecone paid starts at ~$70/mo for 1M vectors.

**What breaks at 10,000 users:**
- Pinecone free tier vector limit (100k) → upgrade or migrate to Qdrant
- `all-MiniLM` CPU bottleneck → switch to Groq's hosted embeddings or batch GPU
- No reranking → add a cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) for better precision

---

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ (or use `npm create vite@5` for Node 20.9)
- [Pinecone account](https://pinecone.io) — free tier
- [Groq account](https://console.groq.com) — free tier

### Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp ../.env.example .env
# Edit .env and add your keys:
#   PINECONE_API_KEY=your_key
#   GROQ_API_KEY=your_key

# Start server
uvicorn main:app --reload
# Running on http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Running on http://localhost:5173
```

### Environment Variables

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## Project Structure

```
creatorjoy/
├── backend/
│   ├── main.py          # FastAPI app — /ingest and /chat/stream routes
│   ├── ingestion.py     # YouTube transcript fetch + parent-child chunking
│   ├── embeddings.py    # all-MiniLM-L6-v2 + Pinecone upsert
│   ├── retrieval.py     # Pinecone query + parent deduplication + context building
│   ├── chat.py          # Groq streaming + SSE token delivery
│   ├── models.py        # Pydantic schemas (VideoData, ChildChunk, ParentChunk, etc.)
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── App.jsx                        # Two-panel layout
│       ├── api/index.js                   # ingestVideos() + streamChat()
│       └── components/
│           ├── VideoInput.jsx             # URL inputs + thumbnail preview
│           ├── ChatPanel.jsx              # Streaming chat + suggested prompts
│           └── SourceCitations.jsx        # Citation cards with timestamps + scores
│
├── .env.example
└── README.md
```

---

## API Reference

### `POST /ingest`

```json
// Request
{
  "url_a": "https://www.youtube.com/watch?v=...",
  "url_b": "https://www.youtube.com/watch?v=..."
}

// Response
{
  "session_id": "uuid-v4",
  "video_a": { "title": "...", "author": "...", "thumbnail_url": "...", ... },
  "video_b": { ... },
  "message": "Ingested 48 chunks into Pinecone."
}
```

### `POST /chat/stream`

```json
// Request
{
  "session_id": "uuid-v4",
  "message": "Why did Video A outperform Video B?",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}

// SSE Response stream
data: {"type": "sources", "sources": [...]}
data: {"type": "token", "content": "Based"}
data: {"type": "token", "content": " on"}
...
data: {"type": "done"}
```

---

## Key Design Decisions

**Why no YouTube Data API?**
The oEmbed endpoint gives us title, author, and thumbnail without requiring an API key. For engagement metrics (views, likes), YouTube Data API v3 would be the production path — intentionally excluded here to keep the setup zero-credential for transcript access.

**Why session namespaces in Pinecone?**
Each ingest call creates a UUID session. All vectors for that session live under that namespace. Multiple users never bleed into each other, and cleanup is trivial — drop the namespace.

**Why pass full history per turn instead of a memory store?**
Works perfectly up to ~10 turns (the typical creator session). Simpler than LangChain memory, zero infra, and fully transparent to debug. At scale, the right move is summarising history older than 5 turns with a cheap model call.

**Why Groq over OpenAI?**
Free tier is fast enough for demos (inference is typically under 2 seconds for a 1k-token response). llama-3.3-70b-versatile has strong reasoning for content analysis. Swapping to GPT-4o is a one-line model string change.

---

## What I'd Build Next

1. **YouTube Data API integration** — real view/like/comment counts for engagement rate computation
2. **Cross-encoder reranking** — better precision on retrieval before sending to LLM  
3. **Session persistence** — save sessions to a DB so creators can return to previous comparisons
4. **Multi-platform support** — TikTok, Instagram Reels via `yt-dlp` for transcript extraction
5. **Hook scoring** — dedicated pipeline that scores the first 30 seconds of each video against proven hook patterns