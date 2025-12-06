import os
import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from supabase import create_client

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = "claims-index"
EMBED_MODEL = "text-embedding-3-small"
GPT_RAG_MODEL = "gpt-4.1-mini"
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

class RAGSystem:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(INDEX_NAME)
        
        # Init Supabase
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.supabase = create_client(url, key)

    def embed_text(self, text: str) -> List[float]:
        resp = self.openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
        return resp.data[0].embedding

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from Pinecone, then fetch actual text from Supabase.
        """
        q_emb = self.embed_text(query)
        results = self.index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=True,
        )

        # 1. Collect all pair IDs from Pinecone results
        pinecone_matches = []
        all_pair_indices = []
        
        for m in results.matches:
            md = m.metadata or {}
            pair_ids = md.get("pair_ids", [])
            
            # Extract indices for Supabase lookup
            # pair_id format: "{video_id}_{index:03d}"
            # We assume ingestion used 0 for non-numeric video_ids, so ID = index.
            indices = []
            for pid in pair_ids:
                try:
                    # Get last part after underscore
                    idx_str = pid.split("_")[-1]
                    indices.append(int(idx_str))
                except ValueError:
                    continue
            
            all_pair_indices.extend(indices)
            pinecone_matches.append({
                "score": m.score,
                "video_id": md.get("video_id"),
                "pair_ids": pair_ids,
                "indices": indices
            })

        # 2. Batch fetch from Supabase
        db_pairs_map = {}
        if all_pair_indices:
            try:
                # Remove duplicates
                unique_indices = list(set(all_pair_indices))
                res = self.supabase.table("claim_response_pairs") \
                    .select("*") \
                    .in_("id", unique_indices) \
                    .execute()
                
                for row in res.data:
                    db_pairs_map[row['id']] = row
            except Exception as e:
                print(f"Error fetching from Supabase: {e}")

        # 3. Construct context items with Supabase text
        context_items = []
        for pm in pinecone_matches:
            pairs_data = []
            for idx, pid in zip(pm["indices"], pm["pair_ids"]):
                if idx in db_pairs_map:
                    row = db_pairs_map[idx]
                    pairs_data.append({
                        "pair_id": pid,
                        "claim": row.get("claim"),
                        "response": row.get("response")
                    })
            
            if pairs_data:
                context_items.append({
                    "score": pm["score"],
                    "video_id": pm["video_id"],
                    "pair_ids": pm["pair_ids"],
                    "pairs": pairs_data,
                    "text": "" # Not used directly, we use pairs
                })
                
        return context_items

    def generate_answer(self, query: str, context_items: List[Dict[str, Any]]) -> str:
        """
        Generate an answer with inline citations.
        """
        system_prompt = """You are a helpful assistant answering questions about Islamic theology based on the provided video transcripts.
Rules:
1. Answer the user's question using ONLY the provided context.
2. If the answer is not in the context, say "I cannot answer this based on the available information."
3. You MUST cite your sources using the format [video_id:pair_id] immediately after the claim.
   Example: "The speaker argues that X is true [123:1_005], and further explains Y [123:1_006]."
4. Do not make up citations. Use the exact IDs provided in the context.
"""

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
        citation_pattern = r"\[([a-zA-Z0-9_]+):([a-zA-Z0-9_]+)\]"
        matches = re.findall(citation_pattern, answer)
        
        if not matches:
            return [], 1.0 if "I cannot answer" in answer else 0.0

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
                verify_prompt = f"""
You are a fact-checking assistant.
Source Text ({pid}):
{id_to_content[pid]}

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

    def filter_context(self, query: str, context_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use an LLM to filter out irrelevant context items.
        Returns a list of only the relevant items.
        """
        if not context_items:
            return []

        # Prepare prompt
        items_str = ""
        for i, item in enumerate(context_items):
            # Extract text representation
            text = ""
            for p in item.get("pairs", []):
                text += f"Claim: {p.get('claim')}\nResponse: {p.get('response')}\n"
            if not text:
                text = item.get("text", "")
            
            items_str += f"Item {i}:\n{text}\n---\n"

        prompt = f"""
You are a relevance classifier.
User Query: "{query}"

Below is a list of retrieved context items.
Identify which items are relevant to answering the query.
Return the indices of relevant items as a JSON list of integers.
If none are relevant, return [].

Items:
{items_str}

Output JSON:
"""
        try:
            resp = self.openai_client.chat.completions.create(
                model=GPT_RAG_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(resp.choices[0].message.content)
            indices = data.get("indices", [])
            if not isinstance(indices, list):
                # Handle case where model might return {"relevant_indices": ...} or similar if it hallucinates key
                # But with json mode and simple prompt it's usually stable.
                # Let's try to find any list in values.
                for v in data.values():
                    if isinstance(v, list):
                        indices = v
                        break
            
            relevant_items = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(context_items):
                    relevant_items.append(context_items[idx])
            
            return relevant_items
        except Exception as e:
            print(f"Filtering failed: {e}")
            return context_items # Fallback to all items

    def query(self, user_query: str) -> RAGResponse:
        start_time = time.time()
        
        # 1. Retrieve
        raw_context = self.retrieve(user_query, top_k=10) # Fetch more to filter
        
        # 2. Rerank / Filter (Abstention on weak evidence)
        filtered_context = self.filter_context(user_query, raw_context)
        
        if not filtered_context:
            return RAGResponse(
                answer="I cannot answer this based on the available information (No relevant context found).",
                citations=[],
                context_used=[],
                faithfulness_score=1.0, # Technically faithful to the "no info" state
                latency_seconds=time.time() - start_time
            )

        # 3. Generate
        answer = self.generate_answer(user_query, filtered_context)
        
        # 4. Verify
        citations, score = self.verify_citations(answer, filtered_context)
        
        # Post-hoc Abstention: If score is very low and answer is not a refusal
        if score < 0.3 and "I cannot answer" not in answer:
             answer += "\n\n[Warning: The citations provided could not be fully verified against the source text.]"

        latency = time.time() - start_time
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            context_used=filtered_context,
            faithfulness_score=score,
            latency_seconds=latency
        )
