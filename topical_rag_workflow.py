import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from supabase import create_client

# Load environment variables
load_dotenv()

# Configuration
INDEX_NAME = "claims-index"
NAMESPACE = "topical-answers"
EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4.1-mini"
VERIFICATION_MODEL = "gpt-4.1-mini"

@dataclass
class TopicalCitation:
    video_id: str
    timestamp_start: str
    timestamp_end: str
    reasoning: str = ""
    is_faithful: bool = False

@dataclass
class TopicalResult:
    answer: str
    citations: List[TopicalCitation]
    sources: List[Dict[str, Any]]
    faithfulness_score: float
    latency: float

class TopicalRAG:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Ensure index exists
        if not self.pc.has_index(INDEX_NAME):
            self.pc.create_index(
                name=INDEX_NAME,
                vector_type="dense",
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = self.pc.Index(INDEX_NAME)
        
        # Supabase
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.supabase = create_client(url, key)

    def embed_text(self, text: str) -> List[float]:
        resp = self.openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
        return resp.data[0].embedding

    def ingest(self):
        print("\n=== STARTING INGESTION ===")
        print("Fetching data from 'topical_answers'...")
        
        res = self.supabase.table("topical_answers").select("*").execute()
        rows = res.data
        if not rows:
            print("No data found.")
            return

        print(f"Found {len(rows)} rows. Generating embeddings...")
        
        vectors = []
        for i, row in enumerate(rows):
            content = f"Topic: {row['topic']}\nAnswer: {row['answer']}"
            embedding = self.embed_text(content)
            
            metadata = {
                "id": row['id'],
                "video_id": row['video_id'],
                "topic": row['topic'],
                "answer": row['answer'],
                "timestamp_start": row['timestamp_start'],
                "timestamp_end": row['timestamp_end'],
                "topic_num": row['topic_num']
            }
            
            vectors.append({
                "id": row['id'],
                "values": embedding,
                "metadata": metadata
            })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(rows)}...")

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=NAMESPACE)
            print(f"Upserted batch {i}-{i+len(batch)} to namespace '{NAMESPACE}'")
            
        print("=== INGESTION COMPLETE ===\n")

    def verify_citations(self, answer: str, sources: List[Dict[str, Any]]) -> Tuple[List[TopicalCitation], float]:
        """
        Verify citations in the format [video_id:start_time-end_time].
        """
        # Regex to capture [video_id:start-end]
        # Assuming video_id is UUID or string, timestamps are strings.
        # Example: [abc-123:10:00-12:00]
        # Let's make it flexible: [ID:Time]
        citation_pattern = r"\[([a-zA-Z0-9-]+):([^\]]+)\]"
        matches = re.findall(citation_pattern, answer)
        
        if not matches:
             return [], 1.0 if "I cannot answer" in answer else 0.0

        # Map video_id to source content
        vid_to_source = {}
        for s in sources:
            vid = s.get("video_id")
            if vid:
                vid_to_source[vid] = s

        citations = []
        verified_count = 0

        for vid, time_range in matches:
            # Parse start/end from time_range "start-end"
            if "-" in time_range:
                ts_start, ts_end = time_range.split("-", 1)
            else:
                ts_start, ts_end = time_range, ""

            cit = TopicalCitation(video_id=vid, timestamp_start=ts_start, timestamp_end=ts_end)
            
            if vid not in vid_to_source:
                cit.reasoning = "Video ID not found in retrieved context."
                cit.is_faithful = False
            else:
                source = vid_to_source[vid]
                source_text = f"Topic: {source['topic']}\nAnswer: {source['answer']}"
                
                verify_prompt = f"""
You are a fact-checking assistant.
Source Text (Video {vid}):
{source_text}

Generated Answer Fragment containing citation:
"...{answer}..."

Task: Does the Source Text support the statements in the Answer that rely on this citation?
Return JSON: {{"supported": true/false, "reasoning": "..."}}
"""
                try:
                    v_resp = self.openai_client.chat.completions.create(
                        model=VERIFICATION_MODEL,
                        messages=[{"role": "user", "content": verify_prompt}],
                        response_format={"type": "json_object"}
                    )
                    v_data = json.loads(v_resp.choices[0].message.content)
                    cit.is_faithful = v_data.get("supported", False)
                    cit.reasoning = v_data.get("reasoning", "")
                except Exception as e:
                    cit.is_faithful = False
                    cit.reasoning = f"Verification failed: {str(e)}"

            citations.append(cit)
            if cit.is_faithful:
                verified_count += 1

        score = verified_count / len(citations) if citations else 0.0
        return citations, score

    def query(self, user_query: str, top_k: int = 3) -> TopicalResult:
        start_time = time.time()
        
        # 1. Retrieve
        q_emb = self.embed_text(user_query)
        results = self.index.query(
            vector=q_emb,
            top_k=top_k,
            namespace=NAMESPACE,
            include_metadata=True
        )
        
        sources = []
        context_str = ""
        
        for m in results.matches:
            md = m.metadata
            sources.append(md)
            context_str += f"""
Video ID: {md['video_id']}
Timestamps: {md['timestamp_start']} - {md['timestamp_end']}
Topic: {md['topic']}
Answer: {md['answer']}
---
"""

        # 2. Generate
        system_prompt = """You are a helpful assistant. Answer the user's question using the provided context.
Rules:
1. Answer using ONLY the provided context.
2. If the answer is not in the context, say "I cannot answer this based on the available information."
3. You MUST cite your sources using the format [video_id:start_timestamp-end_timestamp] immediately after the claim.
   Example: "The speaker discusses evolution [abc-123:10:00-12:30]."
   Use the exact Video ID and Timestamps provided in the context.
"""
        user_prompt = f"Question: {user_query}\n\nContext:\n{context_str}"
        
        resp = self.openai_client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        answer = resp.choices[0].message.content
        
        # 3. Verify
        citations, score = self.verify_citations(answer, sources)
        
        latency = time.time() - start_time
        return TopicalResult(
            answer=answer, 
            citations=citations, 
            sources=sources, 
            faithfulness_score=score, 
            latency=latency
        )

if __name__ == "__main__":
    rag = TopicalRAG()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        rag.ingest()
    else:
        print("Usage: python topical_rag_workflow.py [ingest]")
        print("Running test query...\n")
        
        q = "What is the speaker's view on evolution?"
        print(f"Query: {q}")
        
        res = rag.query(q)
        print("\n=== ANSWER ===")
        print(res.answer)
        print(f"\nFaithfulness: {res.faithfulness_score:.2f}")
        print("\n=== CITATIONS ===")
        for c in res.citations:
            print(f"- [{c.video_id}:{c.timestamp_start}-{c.timestamp_end}] Faithful: {c.is_faithful}")

