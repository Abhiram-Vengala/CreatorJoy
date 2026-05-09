import re
import httpx
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from typing import Optional
from models import VideoData, ParentChunk, ChildChunk

# ─── Chunking Constants ────────────────────────────────────────────────────────
#
# Parent-Child strategy:
#
#   PARENT (~200 words)
#   └── Child 1 (~65 words)   ← embedded into Pinecone
#   └── Child 2 (~65 words)   ← embedded into Pinecone
#   └── Child 3 (~65 words)   ← embedded into Pinecone
#
# Why this split?
#
# Child size (65 words ≈ ~85 tokens):
#   - Close in size to a typical user query ("why did my hook fail in the first 5 secs?")
#   - Pinecone's own developer advocate recommends matching chunk size to query size
#     for higher cosine similarity scores
#   - Small enough for the embedding model to capture ONE focused idea per vector
#
# Parent size (200 words ≈ ~270 tokens):
#   - Big enough to give the LLM full narrative context (before + after the matched child)
#   - 3 parents × 270 tokens = ~810 tokens of context — leaves room for history in 8k window
#   - NOT embedded — only stored as metadata on the child vector in Pinecone
#
# Overlap between parents (30 words):
#   - Prevents context loss at parent boundaries
#   - Small because child overlap already handles fine-grained boundary issues
#
# At 1000 creators/day with avg 10-min video (~1500 words):
#   - ~8 parents per video → ~24 child vectors per video
#   - 2 videos per session → 48 vectors per session
#   - 1000 sessions → 48,000 vectors/day — well within Pinecone free tier (100k vectors)

PARENT_WORDS = 200
PARENT_OVERLAP = 30
CHILDREN_PER_PARENT = 3  # parent is split into 3 equal children


def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:shorts\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


async def fetch_video_metadata(video_id: str) -> dict:
    """
    Fetch video metadata using YouTube oEmbed API (no API key needed).
    oEmbed doesn't expose view/like counts — would need YouTube Data API v3 for that.
    """
    oembed_url = (
        f"https://www.youtube.com/oembed"
        f"?url=https://www.youtube.com/watch?v={video_id}&format=json"
    )
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(oembed_url)
        resp.raise_for_status()
        data = resp.json()

    return {
        "title": data.get("title", "Unknown Title"),
        "author": data.get("author_name", "Unknown Channel"),
        "thumbnail_url": data.get("thumbnail_url", ""),
        "view_count": None,
        "like_count": None,
        "duration_seconds": None,
        "engagement_rate": None,
    }


def fetch_transcript(video_id: str) -> list[dict]:
    """Fetch raw transcript from YouTube. Returns list of {text, start, duration}."""
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id)
        return [
            {"text": s.text, "start": s.start, "duration": s.duration}
            for s in fetched
        ]
    except TranscriptsDisabled:
        raise ValueError(f"Transcripts are disabled for video {video_id}")
    except NoTranscriptFound:
        raise ValueError(f"No transcript found for video {video_id}")
    except Exception as e:
        raise ValueError(f"Failed to fetch transcript: {str(e)}")


def build_parent_child_chunks(
    transcript: list[dict],
    video_id: str,
    video_title: str,
) -> list[ParentChunk]:
    """
    Two-level chunking pipeline:

    Step 1 — Build parents (200 words, 30-word overlap)
      Each parent captures one broad topic segment of the video.

    Step 2 — Split each parent into N equal children
      Children are what get embedded. They're smaller and closer in
      size to user queries, improving cosine similarity at retrieval time.
      The parent text is stored in each child's metadata so we can
      return rich context to the LLM without a second lookup.

    Step 3 — Attach real timestamps
      Each chunk gets an approximate video timestamp by mapping its
      word position back to the original transcript timeline.
    """
    full_text = " ".join([entry["text"] for entry in transcript])
    words = full_text.split()
    total_words = len(words)

    parent_chunks: list[ParentChunk] = []
    p_start = 0
    p_idx = 0

    while p_start < total_words:
        p_end = min(p_start + PARENT_WORDS, total_words)
        parent_text = " ".join(words[p_start:p_end])
        parent_timestamp = _word_pos_to_timestamp(transcript, p_start, total_words)

        parent_id = f"{video_id}#p{p_idx}"

        # ── Split parent into children ─────────────────────────────────────
        parent_words_list = parent_text.split()
        n_parent_words = len(parent_words_list)
        child_size = max(1, n_parent_words // CHILDREN_PER_PARENT)

        children: list[ChildChunk] = []
        for c_idx in range(CHILDREN_PER_PARENT):
            c_start = c_idx * child_size
            c_end = c_start + child_size if c_idx < CHILDREN_PER_PARENT - 1 else n_parent_words
            child_text = " ".join(parent_words_list[c_start:c_end])

            if not child_text.strip():
                continue

            child_word_offset = p_start + c_start
            child_timestamp = _word_pos_to_timestamp(transcript, child_word_offset, total_words)

            children.append(
                ChildChunk(
                    child_id=f"{parent_id}#c{c_idx}",
                    parent_id=parent_id,
                    video_id=video_id,
                    video_title=video_title,
                    child_text=child_text,
                    parent_text=parent_text,   # full parent stored here — no second DB needed
                    child_index=c_idx,
                    parent_index=p_idx,
                    timestamp_approx=child_timestamp,
                )
            )

        parent_chunks.append(
            ParentChunk(
                parent_id=parent_id,
                video_id=video_id,
                video_title=video_title,
                text=parent_text,
                parent_index=p_idx,
                timestamp_approx=parent_timestamp,
                children=children,
            )
        )

        p_start += PARENT_WORDS - PARENT_OVERLAP
        p_idx += 1

    return parent_chunks


def _word_pos_to_timestamp(
    transcript: list[dict], word_pos: int, total_words: int
) -> float:
    """Map a word position to an approximate video timestamp in seconds."""
    if not transcript or total_words == 0:
        return 0.0
    ratio = word_pos / total_words
    total_duration = transcript[-1]["start"] + transcript[-1].get("duration", 0)
    return round(ratio * total_duration, 2)


async def ingest_video(url: str) -> VideoData:
    """
    Full ingestion pipeline for a single YouTube URL.
    Returns VideoData with parent-child chunks ready for embedding.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url}")

    metadata = await fetch_video_metadata(video_id)
    transcript = fetch_transcript(video_id)

    parent_chunks = build_parent_child_chunks(
        transcript=transcript,
        video_id=video_id,
        video_title=metadata["title"],
    )

    return VideoData(
        video_id=video_id,
        url=url,
        title=metadata["title"],
        author=metadata["author"],
        thumbnail_url=metadata["thumbnail_url"],
        view_count=metadata["view_count"],
        like_count=metadata["like_count"],
        duration_seconds=metadata["duration_seconds"],
        engagement_rate=metadata["engagement_rate"],
        parent_chunks=parent_chunks,
        raw_transcript=" ".join([e["text"] for e in transcript]),
    )