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

OUTPUT_FILE = "generated_topical_eval_set.json"
NUM_SAMPLES = 20

def fetch_random_rows(limit: int = 50) -> List[Dict[str, Any]]:
    print("Fetching rows from topical_answers...")
    res = supabase.table("topical_answers").select("*").limit(limit).execute()
    data = res.data
    if not data:
        return []
    
    random.shuffle(data)
    return data[:NUM_SAMPLES]

def generate_question(topic: str, answer: str) -> str:
    prompt = f"""
You are an expert at creating reading comprehension questions.
Given the following topic and answer:

Topic: "{topic}"
Answer: "{answer}"

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
    rows = fetch_random_rows(100)
    print(f"Selected {len(rows)} rows for generation.")
    
    eval_set = []
    
    for i, row in enumerate(rows):
        print(f"[{i+1}/{len(rows)}] Generating question for Topic: {row['topic'][:30]}...")
        question = generate_question(row['topic'], row['answer'])
        
        eval_set.append({
            "query": question,
            "relevant_id": row['id'], # UUID
            "topic": row['topic'],
            "answer": row['answer']
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(eval_set, f, indent=2)
    
    print(f"Saved {len(eval_set)} items to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
