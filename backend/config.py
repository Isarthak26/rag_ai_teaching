"""
Configuration — edit this file with your API keys and preferences
"""

import os

# ── API Keys ───────────────────────────────────────────────────────────────────
# Set these as environment variables OR replace the defaults below
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-your-openai-key-here")

# ── Whisper ────────────────────────────────────────────────────────────────────
# Options: tiny | base | small | medium | large
# Larger = more accurate but slower. "base" is a good balance for demos.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "300"))   # tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50")) # overlap between chunks

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# ── Reranker ──────────────────────────────────────────────────────────────────
# Cross-encoder model for re-ranking retrieved chunks
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI Teaching Assistant that helps students understand lecture content.

You ONLY answer based on the provided lecture transcript context. 
- Always cite the timestamp where the information was found (e.g., "At 00:12:30, the instructor explains...")
- If the answer is not found in the context, say: "I couldn't find this in the lecture content. You may want to check the full video or ask your instructor."
- Keep answers clear, concise, and educational.
- Do not make up information beyond what is in the context.
"""
