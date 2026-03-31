from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid, asyncio, shutil, os, tempfile

from ingestion import ingest_youtube_video, ingest_uploaded_file
from rag_chain import answer_question
from store import job_store, vector_stores

app = FastAPI(title="RAG Teaching Assistant API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class IngestRequest(BaseModel):
    youtube_url: str

class QuestionRequest(BaseModel):
    session_id: str
    question: str
    top_k: Optional[int] = 3

class JobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    session_id: Optional[str] = None
    video_title: Optional[str] = None
    chunk_count: Optional[int] = None

@app.get("/")
def root():
    return {"message": "RAG Teaching Assistant API is running"}

@app.post("/ingest", response_model=JobStatus)
async def ingest(req: IngestRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "pending", "message": "Job queued", "session_id": None, "video_title": None, "chunk_count": None}
    bg.add_task(_run_ingestion, job_id, req.youtube_url)
    return JobStatus(job_id=job_id, status="pending", message="Job queued")

@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(job_id=job_id, **job_store[job_id])

@app.post("/ask")
async def ask(req: QuestionRequest):
    if req.session_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Session not found.")
    return await answer_question(req.session_id, req.question, req.top_k)

@app.get("/sessions")
def list_sessions():
    return {"sessions": list(vector_stores.keys())}

@app.post("/upload", response_model=JobStatus)
async def upload_file(bg: BackgroundTasks, file: UploadFile = File(...)):
    ALLOWED = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".ogg"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    finally:
        tmp.close()
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "pending", "message": "File received", "session_id": None, "video_title": None, "chunk_count": None}
    bg.add_task(_run_file_ingestion, job_id, tmp_path, file.filename)
    return JobStatus(job_id=job_id, status="pending", message="File received")

async def _run_ingestion(job_id: str, url: str):
    job_store[job_id]["status"] = "processing"
    try:
        result = await asyncio.to_thread(ingest_youtube_video, url, job_id)
        job_store[job_id].update({"status": "done", "message": "Done", "session_id": result["session_id"], "video_title": result["video_title"], "chunk_count": result["chunk_count"]})
    except Exception as e:
        job_store[job_id].update({"status": "error", "message": str(e)})

async def _run_file_ingestion(job_id: str, file_path: str, original_name: str):
    job_store[job_id]["status"] = "processing"
    try:
        result = await asyncio.to_thread(ingest_uploaded_file, file_path, original_name, job_id)
        job_store[job_id].update({"status": "done", "message": "Done", "session_id": result["session_id"], "video_title": result["video_title"], "chunk_count": result["chunk_count"]})
    except Exception as e:
        job_store[job_id].update({"status": "error", "message": str(e)})
    finally:
        try: os.remove(file_path)
        except: pass
