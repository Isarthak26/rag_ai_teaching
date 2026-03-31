# RAG-Based AI Teaching Assistant
### SIT Special Topics Project — Working Implementation

A fully functional RAG pipeline that:
1. Takes a **YouTube lecture video URL**
2. Downloads audio with **yt-dlp**
3. Transcribes with **OpenAI Whisper**
4. Chunks + embeds into a **FAISS vector store**
5. Answers student questions with **GPT-4o**, citing timestamps

---

## Project Structure

```
rag_project/
├── backend/
│   ├── main.py          ← FastAPI server (REST API)
│   ├── ingestion.py     ← YouTube download + Whisper + FAISS indexing
│   ├── rag_chain.py     ← Retrieval + reranking + GPT-4o generation
│   ├── config.py        ← API keys, model settings, prompts
│   └── store.py         ← In-memory job + vector store
├── frontend/
│   └── index.html       ← Browser UI (open directly, no build needed)
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Prerequisites

Make sure you have these installed:
- Python 3.10+
- **ffmpeg** (required by Whisper for audio processing)

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

### 2. Clone / copy this project

```bash
cd rag_project
```

### 3. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> Note: Whisper will download the model (~140MB for "base") on first run.

### 5. Set your OpenAI API key

```bash
# Mac/Linux
export OPENAI_API_KEY="sk-your-key-here"

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-your-key-here

# Or create a .env file in the backend/ folder:
echo "OPENAI_API_KEY=sk-your-key-here" > backend/.env
```

> Get your key at: https://platform.openai.com/api-keys

### 6. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 7. Open the frontend

Just open `frontend/index.html` in your browser — no server needed for the UI.

---

## Usage

1. Open `frontend/index.html` in your browser
2. Paste any YouTube lecture URL (e.g., a recorded class, tutorial, talk)
3. Click **Download & Process** — the system will:
   - Download the audio (~30s for a 1hr video)
   - Transcribe with Whisper (~2-5 minutes for "base" model)
   - Chunk, embed, and index into FAISS
4. Once done, type any question about the lecture
5. The answer will include **timestamps** you can click to jump to that moment in the video

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Submit YouTube URL for processing |
| GET | `/status/{job_id}` | Poll ingestion job status |
| POST | `/ask` | Ask a question (requires session_id) |
| GET | `/sessions` | List all indexed sessions |

### Example API call

```bash
# 1. Ingest a video
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"}'

# Response: { "job_id": "abc-123", "status": "pending", ... }

# 2. Poll status
curl http://localhost:8000/status/abc-123

# 3. Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "question": "What is RAG?", "top_k": 3}'
```

---

## Configuration

Edit `backend/config.py` to change:

| Setting | Default | Description |
|---------|---------|-------------|
| `WHISPER_MODEL` | `base` | `tiny` (fast) → `large` (accurate) |
| `CHUNK_SIZE` | `300` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `LLM_MODEL` | `gpt-4o` | Any OpenAI chat model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |

---

## RAG Pipeline Explained

```
YouTube URL
    │
    ▼
yt-dlp ──→ MP3 audio file
    │
    ▼
OpenAI Whisper ──→ Timestamped transcript segments
    │
    ▼
LangChain TextSplitter ──→ 300-token chunks (50 overlap)
    │
    ▼
text-embedding-ada-002 ──→ 1536-dim vectors
    │
    ▼
FAISS index (in-memory)
    │
    ▼  [Query time]
Student question ──→ Embed ──→ FAISS top-2k search
    │
    ▼
Cross-encoder reranker ──→ Top-k reranked chunks
    │
    ▼
GPT-4o (with context + timestamps) ──→ Grounded answer
```

---

## Literature Review

This project is grounded in 20 papers reviewed as part of the SIT Special Topics research:

| # | Paper | Key Contribution |
|---|-------|-----------------|
| 1 | Optimizing RAG in Programming Education | Re-ranking improves open-source RAG |
| 2 | CE-GAITA | Local deployment for privacy |
| 3 | RAG-PRISM | Sentiment-driven adaptive responses |
| 4 | RAG + Code Interpreters | Code execution for STEM Q&A |
| 5 | MARK | Hybrid dense+sparse search |
| 6 | KA-RAG | Knowledge graph + agentic RAG |
| 7 | PRAG-EDU | Grade-level adaptive complexity |
| 8 | HiSem-RAG | Hierarchical semantic indexing |
| 9 | Virtual TA in Statistics | Active learning via RAG |
| 10 | Themis | ML fairness Q&A at scale |
| 11 | NeuroBot TA | RAG in medical education |
| 12 | OwlMentor | Scientific literature engagement |
| 13 | Intent → LLM TA | LangChain modular architecture |
| 14 | RAG Chatbot Survey | 47-paper taxonomy |
| 15 | RAG for Education Survey | 4-stage pipeline mapping |
| 16 | Faculty Perspectives on RAG | First faculty-perspective study |
| 17 | GAITA | Personalized CS tutoring |
| 18 | LLM+RAG for Int'l Students | Reducing information friction |
| 19 | High School Guidance RAG | Beyond FAQ chatbots |
| 20 | Lewis et al. (NeurIPS 2020) | **Foundational RAG paper** |

---

## Team

**Project:** RAG-Based AI Teaching Assistant for Lecture Videos  
**Institution:** SIT (Special Topics)  
**Stack:** Python · FastAPI · Whisper · LangChain · FAISS · GPT-4o
# rag_ai_teaching
