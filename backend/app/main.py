import os
import re
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.ingestion.graph_pipeline import ingest_to_graph
from app.ingestion.loader import clean_documents, load_pdf
from app.ingestion.vector_pipeline import ingest_to_vector_db
from app.retrieval.hybrid import answer_question

app = FastAPI(title="Hybrid RAG + Knowledge Graph")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    """Upload a PDF. It goes through BOTH pipelines: vector DB and knowledge graph."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    # 1. Save the uploaded file to disk
    os.makedirs(settings.upload_dir, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", os.path.splitext(file.filename)[0])
    save_path = os.path.join(settings.upload_dir, f"{stem}.pdf")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 2. Load once, then feed both pipelines
        raw_docs = load_pdf(save_path)
        clean_docs = clean_documents(raw_docs)
        if not clean_docs:
            raise HTTPException(
                status_code=422,
                detail="No text found in this PDF. It may be a scanned image.",
            )

        # Pipeline 1: clean -> chunk -> embed -> Chroma
        n_chunks = ingest_to_vector_db(clean_docs, doc_id=stem)

        # Pipeline 2: entities -> relationships -> normalize -> Neo4j
        graph_stats = ingest_to_graph(raw_docs, doc_id=stem)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {
        "doc_id": stem,
        "pages": len(raw_docs),
        "vector_chunks": n_chunks,
        "graph": graph_stats,
    }


@app.post("/ask")
def ask(req: AskRequest):
    """Ask a question. Uses vector search + graph lookup together."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    try:
        return answer_question(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not answer: {e}")