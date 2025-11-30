import os
import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = "claims-index"
EMBED_MODEL = "text-embedding-3-small"
GPT_RAG_MODEL = "gpt-4.1-mini"  # Using mini for speed/cost, can swap to gpt-4o
VERIFICATION_MODEL = "gpt-4.1-mini"

@dataclass
class Citation:
    video_id: str
    pair_id: str
    reasoning: str = ""
    is_faithful: bool = False

@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation]
    context_used: List[Dict[str, Any]]
    faithfulness_score: float
    latency_seconds: float

class RAGClient:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(INDEX_NAME)

    def embed_text(self, text: str) -> List[float]:
        resp = self.openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
        return resp.data[0].embedding

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from Pinecone.
        Returns a list of context items with metadata.
        """
        q_emb = self.embed_text(query)
        results = self.index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=True,
        )

        context_items = []
        for m in results.matches:
            md = m.metadata or {}
            pair_ids = md.get("pair_ids", [])
            try:
                pairs = json.loads(md.get("pairs_json", "[]"))
            except json.JSONDecodeError:
                pairs = []

            context_items.append({
                "score": m.score,
                "video_id": md.get("video_id"),
                "pair_ids": pair_ids,
                "pairs": pairs,
                "text": md.get("text", "") # Fallback if we want raw text
            })
        return context_items

    def generate_answer(self, query: str, context_items: List[Dict[str, Any]]) -> str:
        """
        Generate an answer with inline citations.
        """
        # Prepare context for the LLM
        # We want to give it structured info so it can cite [video_id:pair_id]
        
        system_prompt = """You are a helpful assistant answering questions about Islamic theology based on the provided video transcripts.
Rules:
1. Answer the user's question using ONLY the provided context.
2. If the answer is not in the context, say "I cannot answer this based on the available information."
3. You MUST cite your sources using the format [video_id:pair_id] immediately after the claim.
   Example: "The speaker argues that X is true [123:1_005], and further explains Y [123:1_006]."
4. Do not make up citations. Use the exact IDs provided in the context.
"""

        # Format context
        context_str = ""
        for item in context_items:
            vid = item.get("video_id", "unknown")
            for p in item.get("pairs", []):
                pid = p.get("pair_id", "unknown")
                claim = p.get("claim", "")
                resp = p.get("response", "")
                context_str += f"Source ID: [{vid}:{pid}]\nClaim: {claim}\nResponse: {resp}\n---\n"

        user_prompt = f"Question: {query}\n\nContext:\n{context_str}"

        response = self.openai_client.chat.completions.create(
            model=GPT_RAG_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content

    def verify_citations(self, answer: str, context_items: List[Dict[str, Any]]) -> Tuple[List[Citation], float]:
        """
        Extract citations and verify them against the context.
        Returns list of Citation objects and a faithfulness score (0.0 to 1.0).
        """
        # 1. Extract citations: [video_id:pair_id]
        # Regex to find [123:1_001] style patterns
        citation_pattern = r"\[([a-zA-Z0-9_]+):([a-zA-Z0-9_]+)\]"
        matches = re.findall(citation_pattern, answer)
        
        if not matches:
            return [], 1.0 if "I cannot answer" in answer else 0.0 # If no citations and not a refusal, low trust? Or maybe it's general knowledge? 
            # For RAG, we usually expect citations. Let's assume 0.0 if it claims facts without sources, unless it's a refusal.
            # Actually, if the model just says "Hello", we don't want to penalize. 
            # Let's keep it simple: No citations = 0.0 score if it looks like an answer.

        # Map pair_id -> content for quick lookup
        id_to_content = {}
        for item in context_items:
            for p in item.get("pairs", []):
                pid = p.get("pair_id")
                if pid:
                    id_to_content[pid] = f"Claim: {p.get('claim')}\nResponse: {p.get('response')}"

        citations = []
        verified_count = 0

        for vid, pid in matches:
            cit = Citation(video_id=vid, pair_id=pid)
            
            if pid not in id_to_content:
                cit.reasoning = "Source ID not found in retrieved context."
                cit.is_faithful = False
            else:
                # Verify with LLM
                # We want to check if the sentence *preceding* the citation is supported by the content.
                # This is hard to do perfectly without sentence segmentation. 
                # For this prototype, we will check if the *Answer* as a whole is supported by the *Cited Chunk*.
                # A more granular approach would be: Extract sentence -> Verify against chunk.
                
                # Let's do a simpler verification: "Does the text associated with ID {pid} support the statements made in the answer?"
                # This might be too broad.
                # Better: "Here is a claim made in the answer which cites {pid}. Is it supported?"
                # But we don't know exactly which part of the answer refers to which citation without parsing.
                
                # PROTOTYPE SHORTCUT: We will assume the citation applies to the immediately preceding statement.
                # We will ask the LLM to verify the specific link.
                
                verify_prompt = f"""
You are a fact-checking assistant.
Source Text ({pid}):
{id_to_content[pid]}

Generated Answer Fragment containing citation:
"...{answer}..."

Task: Does the Source Text support the statements in the Answer that rely on this citation?
Return JSON: {{"supported": true/false, "reasoning": "..."}}
"""
                # Optimisation: Sending the whole answer is lazy but works for short answers.
                
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

    def query(self, user_query: str) -> RAGResponse:
        start_time = time.time()
        
        # 1. Retrieve
        context = self.retrieve(user_query)
        
        # 2. Generate
        answer = self.generate_answer(user_query, context)
        
        # 3. Verify
        citations, score = self.verify_citations(answer, context)
        
        latency = time.time() - start_time
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            context_used=context,
            faithfulness_score=score,
            latency_seconds=latency
        )


