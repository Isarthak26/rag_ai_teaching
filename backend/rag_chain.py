"""
RAG Chain
Query  →  FAISS retrieval  →  Cross-encoder reranking  →  GPT-4o generation
"""

from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder

from store import vector_stores
from config import OPENAI_API_KEY, LLM_MODEL, RERANKER_MODEL, SYSTEM_PROMPT

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Lazy-load cross-encoder reranker
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        print(f"[Reranker] Loading: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


async def answer_question(session_id: str, question: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline for a question.
    Returns: answer, retrieved_chunks, model_used
    """
    session = vector_stores[session_id]
    vectorstore = session["vectorstore"]

    # ── Step 1: Retrieve top-k*2 candidates via FAISS ─────────────────────────
    candidate_k = min(top_k * 2, 8)
    docs_and_scores = vectorstore.similarity_search_with_score(question, k=candidate_k)

    # ── Step 2: Re-rank with cross-encoder ────────────────────────────────────
    reranker = get_reranker()
    pairs = [(question, doc.page_content) for doc, _ in docs_and_scores]
    rerank_scores = reranker.predict(pairs)

    # Sort by reranker score descending, keep top_k
    ranked = sorted(
        zip(docs_and_scores, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # ── Step 3: Build context ──────────────────────────────────────────────────
    context_parts = []
    retrieved_chunks = []

    for (doc, faiss_score), rerank_score in ranked:
        ts = doc.metadata.get("timestamp", "unknown")
        source = doc.metadata.get("source", "Lecture")
        url = doc.metadata.get("youtube_url", "")
        yt_link = f"{url}&t={_ts_to_seconds(ts)}s" if url else ""

        context_parts.append(
            f"[Source: {source} | Timestamp: {ts}]\n{doc.page_content}"
        )
        retrieved_chunks.append({
            "text": doc.page_content,
            "timestamp": ts,
            "source": source,
            "youtube_link": yt_link,
            "faiss_score": round(float(faiss_score), 4),
            "rerank_score": round(float(rerank_score), 4),
        })

    context = "\n\n---\n\n".join(context_parts)

    # ── Step 4: Generate answer with GPT-4o ───────────────────────────────────
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context from lecture (with timestamps):\n\n{context}\n\n"
                f"---\n\nStudent Question: {question}\n\n"
                "Answer based ONLY on the provided lecture context. "
                "Always mention the timestamp(s) where the information was found."
            )
        }
    ]

    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=600,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "model_used": LLM_MODEL,
        "chunks_retrieved": len(retrieved_chunks),
        "video_title": session["video_title"],
    }


def _ts_to_seconds(ts: str) -> int:
    """Convert HH:MM:SS to seconds (for YouTube deep-links)."""
    try:
        parts = ts.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        pass
    return 0
