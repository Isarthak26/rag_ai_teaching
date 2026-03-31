"""
Ingestion Pipeline
YouTube URL  →  yt-dlp (audio)  →  Whisper (transcript)  →  Chunks  →  FAISS
Uploaded file →  ffmpeg (audio extract)  →  Whisper  →  Chunks  →  FAISS
"""

import os
import uuid
import tempfile
import subprocess
import yt_dlp
import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from store import vector_stores, job_store
from config import OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, WHISPER_MODEL

# Lazy-load Whisper model (loaded once, reused)
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print(f"[Whisper] Loading model: {WHISPER_MODEL}")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


def ingest_youtube_video(youtube_url: str, job_id: str) -> dict:
    """
    Full ingestion pipeline for a YouTube video.
    Returns session_id, video_title, chunk_count.
    """
    session_id = str(uuid.uuid4())

    # ── Step 1: Download audio ─────────────────────────────────────────────────
    _update_job(job_id, "Downloading audio from YouTube...")
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path, video_title = _download_audio(youtube_url, tmpdir)

        # ── Step 2: Transcribe with Whisper ────────────────────────────────────
        _update_job(job_id, f"Transcribing: '{video_title}' with Whisper ({WHISPER_MODEL})...")
        transcript_segments = _transcribe(audio_path)

    # ── Step 3: Build documents with timestamps ────────────────────────────────
    _update_job(job_id, "Chunking transcript...")
    docs = _build_documents(transcript_segments, video_title, youtube_url)

    # ── Step 4: Embed + store in FAISS ────────────────────────────────────────
    _update_job(job_id, f"Embedding {len(docs)} chunks and building FAISS index...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Persist in memory (keyed by session_id)
    vector_stores[session_id] = {
        "vectorstore": vectorstore,
        "video_title": video_title,
        "youtube_url": youtube_url,
        "chunk_count": len(docs),
    }

    print(f"[Ingestion] Done. session_id={session_id}, chunks={len(docs)}")
    return {
        "session_id": session_id,
        "video_title": video_title,
        "chunk_count": len(docs),
    }


def ingest_uploaded_file(file_path: str, original_name: str, job_id: str) -> dict:
    """
    Ingestion pipeline for a user-uploaded video or audio file.
    Supports: mp4, mkv, avi, mov, webm, mp3, wav, m4a, ogg
    Returns session_id, video_title, chunk_count.
    """
    session_id = str(uuid.uuid4())
    title = os.path.splitext(original_name)[0]  # filename without extension
    ext = os.path.splitext(file_path)[1].lower()

    # ── Step 1: Extract audio if it's a video file ─────────────────────────────
    AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg"}
    if ext in AUDIO_EXTS:
        audio_path = file_path  # already audio, use directly
        _update_job(job_id, f"Audio file received: '{title}'")
    else:
        _update_job(job_id, f"Extracting audio from '{title}'...")
        audio_path = file_path.rsplit(".", 1)[0] + "_audio.mp3"
        _extract_audio_ffmpeg(file_path, audio_path)

    # ── Step 2: Transcribe with Whisper ────────────────────────────────────────
    _update_job(job_id, f"Transcribing '{title}' with Whisper ({WHISPER_MODEL})...")
    transcript_segments = _transcribe(audio_path)

    # Clean up extracted audio if it was a separate file
    if audio_path != file_path and os.path.exists(audio_path):
        os.remove(audio_path)

    # ── Step 3: Chunk ──────────────────────────────────────────────────────────
    _update_job(job_id, "Chunking transcript...")
    docs = _build_documents(transcript_segments, title, url=None)

    # ── Step 4: Embed + FAISS ──────────────────────────────────────────────────
    _update_job(job_id, f"Embedding {len(docs)} chunks and building FAISS index...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)

    vector_stores[session_id] = {
        "vectorstore": vectorstore,
        "video_title": title,
        "youtube_url": None,
        "chunk_count": len(docs),
    }

    print(f"[Ingestion/Upload] Done. session_id={session_id}, chunks={len(docs)}")
    return {
        "session_id": session_id,
        "video_title": title,
        "chunk_count": len(docs),
    }




def _download_audio(url: str, output_dir: str) -> tuple[str, str]:
    """Download audio track using yt-dlp. Returns (file_path, video_title)."""
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Unknown Video")
        # Find the downloaded mp3
        safe_title = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp3"
        return safe_title, title


def _transcribe(audio_path: str) -> list[dict]:
    """
    Transcribe audio with OpenAI Whisper.
    Returns list of segments: [{start, end, text}, ...]
    """
    model = get_whisper_model()
    result = model.transcribe(audio_path, fp16=False, verbose=False)
    return result["segments"]  # each has: id, start, end, text


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _extract_audio_ffmpeg(input_path: str, output_path: str):
    """Use ffmpeg to extract audio from a video file to mp3."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",           # no video stream
        "-ar", "16000",  # 16kHz (optimal for Whisper)
        "-ac", "1",      # mono
        "-b:a", "128k",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")


def _build_documents(segments: list[dict], title: str, url: str = None) -> list[Document]:
    """
    Convert Whisper segments into LangChain Documents with metadata,
    then split into overlapping chunks while preserving timestamp info.
    """
    # First, join segments into one big text with embedded timestamps
    full_text = ""
    for seg in segments:
        ts = _format_timestamp(seg["start"])
        full_text += f"[{ts}] {seg['text'].strip()} "

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    raw_chunks = splitter.split_text(full_text)

    # Build LangChain Documents with metadata
    docs = []
    for i, chunk in enumerate(raw_chunks):
        # Extract timestamp from the start of the chunk if present
        ts = "00:00:00"
        if chunk.startswith("["):
            try:
                ts = chunk[1:chunk.index("]")]
            except ValueError:
                pass

        docs.append(Document(
            page_content=chunk,
            metadata={
                "source": title,
                "youtube_url": url,
                "chunk_index": i,
                "timestamp": ts,
            }
        ))
    return docs


def _update_job(job_id: str, message: str):
    if job_id in job_store:
        job_store[job_id]["message"] = message
    print(f"[Job {job_id[:8]}] {message}")
