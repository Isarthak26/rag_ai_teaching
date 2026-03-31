"""
In-memory stores for job tracking and vector stores.
For production: replace with Redis (jobs) and a persistent vector DB (FAISS/Pinecone).
"""

# job_id -> { status, message, session_id, video_title, chunk_count }
job_store: dict = {}

# session_id -> { vectorstore, video_title, youtube_url, chunk_count }
vector_stores: dict = {}
