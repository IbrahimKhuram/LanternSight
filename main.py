from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from rag_engine import RAGClient, RAGResponse

app = FastAPI(title="LanternSight RAG API", version="0.1.0")

# Initialize RAG Client (lazy load or on startup)
rag_client = None

@app.on_event("startup")
def startup_event():
    global rag_client
    rag_client = RAGClient()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class CitationModel(BaseModel):
    video_id: str
    pair_id: str
    reasoning: str
    is_faithful: bool

class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationModel]
    faithfulness_score: float
    latency_seconds: float
    context_used: Optional[List[dict]] = None # Optional for debugging

@app.get("/")
def read_root():
    return {"status": "online", "message": "LanternSight RAG API is running."}

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if not rag_client:
        raise HTTPException(status_code=503, detail="RAG Client not initialized")
    
    try:
        result = rag_client.query(request.query)
        
        # Convert internal Citation objects to Pydantic models
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
            context_used=result.context_used # Include for transparency in prototype
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
