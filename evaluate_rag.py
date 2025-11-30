import time
import json
from typing import List, Dict
from rag_engine import RAGClient

# Dev set from RAG_Chunking_Response.py (subset or full)
DEV_QUERIES = [
    {
        "query": "Why is Hinduism not right?",
        "relevant_pair_ids": ["3_001", "3_002"],
    },
    {
        "query": "Why is the Quran followed so exactly?",
        "relevant_pair_ids": ["3_004", "3_005"],
    },
    {
        "query": "What does the Muslim speaker say about morality depending on culture and organized religion?",
        "relevant_pair_ids": ["3_003"],
    },
    {
        "query": "Why does the Muslim speaker say the Quran is preserved and valid until the end of times?",
        "relevant_pair_ids": ["3_004"],
    },
    {
        "query": "How does the Muslim speaker compare Muslims following the Quran to Christians following the Bible?",
        "relevant_pair_ids": ["3_005"],
    },
    {
        "query": "How does the Muslim speaker respond to the idea that Buddha could be like a prophet?",
        "relevant_pair_ids": ["2_002"],
    },
    {
        "query": "Is all the information and evidences for Islam contained in the Quran?",
        "relevant_pair_ids": ["2_003"],
    },
    {
        "query": "What does the Muslim speaker say happens to people who never received the message in this life?",
        "relevant_pair_ids": ["1_015"],
    },
    {
        "query": "What advice does the Muslim speaker give about reading the Quran?",
        "relevant_pair_ids": ["1_016"],
    },
    {
        "query": "How does the Muslim speaker answer the claim that parts of the Quran might be outdated or ancient?",
        "relevant_pair_ids": ["3_004"],
    },
    {
        "query": "Why does the Muslim speaker argue that Prophet Muhammad cannot just be an enlightened person like Buddha?",
        "relevant_pair_ids": ["2_002"],
    },
    {
        "query": "What evidences and prophecies does the Muslim speaker mention as reasons to believe Islam is true?",
        "relevant_pair_ids": ["2_003"],
    },
]

def run_evaluation():
    print("Initializing RAG Client...")
    client = RAGClient()
    
    total_faithfulness = 0.0
    total_latency = 0.0
    total_queries = len(DEV_QUERIES)
    
    results = []
    
    print(f"\nStarting evaluation on {total_queries} queries...\n")
    
    for i, item in enumerate(DEV_QUERIES):
        query = item["query"]
        print(f"[{i+1}/{total_queries}] Query: {query}")
        
        try:
            res = client.query(query)
            
            total_faithfulness += res.faithfulness_score
            total_latency += res.latency_seconds
            
            print(f"  -> Latency: {res.latency_seconds:.2f}s")
            print(f"  -> Faithfulness: {res.faithfulness_score:.2f}")
            print(f"  -> Citations: {len(res.citations)}")
            
            results.append({
                "query": query,
                "answer": res.answer,
                "faithfulness": res.faithfulness_score,
                "latency": res.latency_seconds,
                "citations": [
                    {"id": f"{c.video_id}:{c.pair_id}", "faithful": c.is_faithful} 
                    for c in res.citations
                ]
            })
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            
    avg_faithfulness = total_faithfulness / total_queries if total_queries > 0 else 0
    avg_latency = total_latency / total_queries if total_queries > 0 else 0
    
    print("\n=== EVALUATION REPORT ===")
    print(f"Total Queries: {total_queries}")
    print(f"Average Faithfulness: {avg_faithfulness:.2%}")
    print(f"Average Latency: {avg_latency:.2f} seconds")
    
    # Optional: Save detailed results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to evaluation_results.json")

if __name__ == "__main__":
    run_evaluation()
