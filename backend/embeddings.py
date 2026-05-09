import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from models import ChildChunk, VideoData

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "creatorjoy"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# Load once at module level — model stays in memory across requests
_model = SentenceTransformer("all-MiniLM-L6-v2")
_pc = Pinecone(api_key=PINECONE_API_KEY)


def get_or_create_index():
    """
    Get or create the Pinecone serverless index.

    Why cosine metric?
    - all-MiniLM-L6-v2 produces normalized vectors
    - Cosine similarity is standard for semantic search on normalized embeddings
    - Dot product would be equivalent here but cosine is more portable if we swap models
    """
    existing = [idx.name for idx in _pc.list_indexes()]
    if INDEX_NAME not in existing:
        _pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return _pc.Index(INDEX_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed a list of strings. Returns list of float vectors."""
    embeddings = _model.encode(texts, batch_size=32, show_progress_bar=False)
    return embeddings.tolist()


def upsert_child_chunks(
    index,
    children: list[ChildChunk],
    session_id: str,
) -> int:
    """
    Embed child chunks and upsert to Pinecone.

    Key design:
    - We embed child_text (small, ~65 words) → better query-chunk size matching
    - We store parent_text in metadata → LLM gets full context without a second lookup
    - ID pattern follows Pinecone's recommended parentId#chunkId prefix scheme
      so sibling chunks and parent lookups are easy later

    Metadata stored per vector:
      video_id, video_title, child_text, parent_text, parent_id,
      child_index, parent_index, timestamp_approx

    Note: Pinecone metadata is not indexed for vector search — it's retrieved
    alongside the match. Storing parent_text here is safe and avoids needing
    a separate docstore (Redis, DynamoDB, etc.) at this scale.
    """
    if not children:
        return 0

    texts = [c.child_text for c in children]
    vectors = embed_texts(texts)

    pinecone_vectors = []
    for child, vector in zip(children, vectors):
        pinecone_vectors.append(
            {
                "id": child.child_id,
                "values": vector,
                "metadata": {
                    "video_id": child.video_id,
                    "video_title": child.video_title,
                    "child_text": child.child_text,
                    "parent_text": child.parent_text,   # ← the key addition
                    "parent_id": child.parent_id,
                    "child_index": child.child_index,
                    "parent_index": child.parent_index,
                    "timestamp_approx": child.timestamp_approx,
                },
            }
        )

    # Batch upserts — Pinecone free tier handles up to 100 vectors per upsert call
    batch_size = 100
    for i in range(0, len(pinecone_vectors), batch_size):
        index.upsert(vectors=pinecone_vectors[i : i + batch_size], namespace=session_id)

    return len(pinecone_vectors)


def embed_and_store(video_a: VideoData, video_b: VideoData, session_id: str) -> dict:
    """
    Full pipeline: flatten all child chunks from both videos → embed → upsert.
    Returns a summary dict for logging/response.
    """
    index = get_or_create_index()

    # Flatten: VideoData → ParentChunks → ChildChunks
    children_a = [child for parent in video_a.parent_chunks for child in parent.children]
    children_b = [child for parent in video_b.parent_chunks for child in parent.children]

    count_a = upsert_child_chunks(index, children_a, session_id)
    count_b = upsert_child_chunks(index, children_b, session_id)

    return {
        "index": INDEX_NAME,
        "session_id": session_id,
        "video_a_parents": len(video_a.parent_chunks),
        "video_a_children": count_a,
        "video_b_parents": len(video_b.parent_chunks),
        "video_b_children": count_b,
        "total_vectors": count_a + count_b,
    }


def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    return _model.encode([query])[0].tolist()