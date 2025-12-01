from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from rag_core import RAGSystem, RAGResponse
from topical_rag_workflow import TopicalRAG, TopicalResult

app = FastAPI(title="LanternSight RAG API", version="0.3.0")

# Initialize RAG Systems
rag_system = None
topical_rag = None

@app.on_event("startup")
def startup_event():
    global rag_system, topical_rag
    rag_system = RAGSystem()
    topical_rag = TopicalRAG()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class CitationModel(BaseModel):
    video_id: str
    pair_id: str
    reasoning: str
    is_faithful: bool

class TopicalCitationModel(BaseModel):
    video_id: str
    timestamp_start: str
    timestamp_end: str
    reasoning: str
    is_faithful: bool

class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationModel]
    faithfulness_score: float
    latency_seconds: float
    context_used: Optional[List[dict]] = None

class TopicalQueryResponse(BaseModel):
    answer: str
    citations: List[TopicalCitationModel]
    faithfulness_score: float
    latency_seconds: float
    sources: Optional[List[dict]] = None

@app.get("/")
def read_root():
    return {"status": "online", "message": "LanternSight RAG API (V3) is running."}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG System not initialized")
    
    try:
        result = rag_system.query(request.query)
        
        api_citations = [
            CitationModel(
                video_id=c.video_id,
                pair_id=c.pair_id,
                reasoning=c.reasoning,
                is_faithful=c.is_faithful
            ) for c in result.citations
        ]
        
        return QueryResponse(
            answer=result.answer,
            citations=api_citations,
            faithfulness_score=result.faithfulness_score,
            latency_seconds=result.latency_seconds,
            context_used=result.context_used
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/topical", response_model=TopicalQueryResponse)
def query_topical(request: QueryRequest):
    if not topical_rag:
        raise HTTPException(status_code=503, detail="Topical RAG System not initialized")
    
    try:
        result = topical_rag.query(request.query, top_k=request.top_k)
        
        api_citations = [
            TopicalCitationModel(
                video_id=c.video_id,
                timestamp_start=c.timestamp_start,
                timestamp_end=c.timestamp_end,
                reasoning=c.reasoning,
                is_faithful=c.is_faithful
            ) for c in result.citations
        ]
        
        return TopicalQueryResponse(
            answer=result.answer,
            citations=api_citations,
            faithfulness_score=result.faithfulness_score,
            latency_seconds=result.latency,
            sources=result.sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
