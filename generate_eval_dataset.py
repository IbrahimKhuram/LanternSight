import os
import json
import random
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()

# Init clients
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(supabase_url, supabase_key)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_FILE = "generated_eval_set.json"
NUM_SAMPLES = 20

def fetch_random_pairs(limit: int = 50) -> List[Dict[str, Any]]:
    print("Fetching pairs from Supabase...")
    # Fetch a larger batch to sample from
    res = supabase.table("claim_response_pairs").select("*").limit(limit).execute()
    data = res.data
    if not data:
        return []
    
    # Shuffle and pick
    random.shuffle(data)
    return data[:NUM_SAMPLES]

def generate_question(claim: str, response: str) -> str:
    prompt = f"""
You are an expert at creating reading comprehension questions.
Given the following claim and response from a dialogue:

Claim: "{claim}"
Response: "{response}"

Generate a single, specific question that this text answers. 
The question should be answerable by the provided text.
Return ONLY the question string.
"""
    resp = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()

def main():
    pairs = fetch_random_pairs(100)
    print(f"Selected {len(pairs)} pairs for generation.")
    
    eval_set = []
    
    for i, p in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] Generating question for ID {p['id']}...")
        question = generate_question(p['claim'], p['response'])
        
        # We need to construct the pair_id format used in RAG
        # The ID in Supabase is an integer (e.g. 11, 1000001)
        # The RAG system expects string IDs like "video_id_index".
        # However, our current Supabase integration in rag_core.py maps numeric IDs back to text.
        # But wait, rag_core.py retrieves based on Pinecone metadata which has "video_id_index".
        # And then it extracts the integer index to query Supabase.
        
        # So for ground truth, we need the "pair_id" string that Pinecone has.
        # But we don't have that easily from just the Supabase row unless we reverse engineer it.
        # Actually, store_claim_response_pairs in RAG_Chunking_Response.py generated IDs:
        # id_int = vid_num * BASE + pair_index
        # So we can try to reverse it if we know BASE=1_000_000.
        
        # Let's try to reconstruct it.
        # id = vid_num * 1_000_000 + pair_index
        # pair_index = id % 1_000_000
        # vid_num = id // 1_000_000
        
        # But wait, some IDs might be small integers if they came from elsewhere?
        # The sample row I saw earlier had ID=11. 
        # If ID=11, then vid_num=0, pair_index=11.
        # So pair_id = "0_011" (padded to 3 chars? The script used :03d).
        
        # Let's assume standard logic:
        db_id = p['id']
        base = 1_000_000
        vid_num = db_id // base
        pair_idx = db_id % base
        
        # We don't know the exact string format of video_id (it might be a UUID or string).
        # But the ingestion script used:
        # vid_str = str(p.get("video_id", "0"))
        # vid_num = int(vid_str) if vid_str.isdigit() else 0
        
        # If the original video_id was a UUID, vid_num is 0.
        # If it was "123", vid_num is 123.
        
        # This is a bit risky for exact string matching in evaluation.
        # However, rag_core.py's retrieve() logic:
        # extracts indices from Pinecone pair_ids and queries Supabase by ID.
        # So if we verify based on the *content* or just the *Supabase ID*, it's safer.
        
        # But run_eval.py checks:
        # "relevant_pair_ids": ["3_001", ...]
        # And rag_core returns citations with pair_ids.
        
        # Let's simplify: We will store the Supabase ID as the ground truth.
        # And we will update run_eval.py to check if the retrieved citations *map* to this Supabase ID.
        # Or simpler: The generated set will have "relevant_supabase_ids": [123].
        # And we update run_eval to check if any cited pair maps to that ID.
        
        eval_set.append({
            "query": question,
            "relevant_supabase_ids": [db_id],
            "claim": p['claim'],
            "response": p['response']
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(eval_set, f, indent=2)
    
    print(f"Saved {len(eval_set)} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
