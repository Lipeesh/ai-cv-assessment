"""
FastAPI Server for Task 4 — Industrial Safety & Maintenance RAG Assistant.
"""

import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi import Request
from pydantic import BaseModel, Field

from rag_engine import IndustrialSafetyRAGEngine
from dashboard_html import DASHBOARD_HTML

app = FastAPI(
    title="Industrial Safety & Maintenance RAG Assistant",
    description="API for querying plant SOPs, OSHA regulations, PPE standards, and restricted zone safety rules.",
    version="1.0.0"
)

# Initialize engine globally on startup
engine: Optional[IndustrialSafetyRAGEngine] = None


class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the hard hat requirements in zone alpha?")
    top_k: int = Field(3, ge=1, le=10, description="Number of relevant documents to retrieve")


class DocumentItem(BaseModel):
    doc_id: str
    title: str
    category: str
    content: str
    applicable_zones: List[str]
    osha_reference: Optional[str] = None
    similarity_score: Optional[float] = None
    rank: Optional[int] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[DocumentItem]
    latency_ms: float
    top_similarity_score: float


@app.on_event("startup")
def startup_event():
    global engine
    corpus_path = os.getenv("CORPUS_PATH", "/home/ub/Desktop/COMBINATION/ai-cv-assessment/task4/data/industrial_sop_corpus.json")
    model_name = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    engine = IndustrialSafetyRAGEngine(corpus_path=corpus_path, model_name=model_name, device="cpu")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" not in accept and "application/json" in accept:
        return JSONResponse({
            "service": "AjnaLens Industrial Safety RAG API",
            "status": "online",
            "docs_url": "/docs",
            "health_check": "/health"
        })
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/health")
def health():
    if engine is None or engine.doc_embeddings is None:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")
    return {
        "status": "healthy",
        "documents_indexed": len(engine.documents),
        "embedding_dimensions": int(engine.doc_embeddings.shape[1]),
        "device": engine.device
    }


@app.get("/documents")
def list_documents():
    if engine is None:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")
    return {
        "count": len(engine.documents),
        "documents": [
            {
                "doc_id": d["doc_id"],
                "title": d["title"],
                "category": d["category"],
                "applicable_zones": d.get("applicable_zones", []),
                "osha_reference": d.get("osha_reference", "")
            }
            for d in engine.documents
        ]
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = engine.query(request.query, top_k=request.top_k)
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("RAG_PORT", 8088))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
