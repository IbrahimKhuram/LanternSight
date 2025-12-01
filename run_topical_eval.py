import time
import json
import os
from typing import List, Dict
from topical_rag_workflow import TopicalRAG

GENERATED_EVAL_FILE = "generated_topical_eval_set.json"

def run_evaluation():
    print("Initializing Topical RAG System...")
    rag = TopicalRAG()
    
    if not os.path.exists(GENERATED_EVAL_FILE):
        print(f"Error: {GENERATED_EVAL_FILE} not found.")
        return

    with open(GENERATED_EVAL_FILE, "r") as f:
        eval_data = json.load(f)
    
    total_latency = 0.0
    total_recall = 0.0
    total_faithfulness = 0.0
    total_queries = len(eval_data)
    
    results = []
    
    print(f"\nStarting evaluation on {total_queries} queries from {GENERATED_EVAL_FILE}...\n")
    
    for i, item in enumerate(eval_data):
        query = item["query"]
        target_id = item["relevant_id"]
        
        print(f"[{i+1}/{total_queries}] Query: {query}")
        
        try:
            res = rag.query(query)
            
            total_latency += res.latency
            total_faithfulness += res.faithfulness_score
            
            # Check Recall: Did we retrieve the chunk that generated the question?
            hit = False
            for src in res.sources:
                if src.get("id") == target_id:
                    hit = True
                    break
            
            if hit:
                total_recall += 1.0
            
            print(f"  -> Latency: {res.latency:.2f}s")
            print(f"  -> Faithfulness: {res.faithfulness_score:.2f}")
            print(f"  -> Target Hit: {hit}")
            
            results.append({
                "query": query,
                "answer": res.answer,
                "latency": res.latency,
                "faithfulness": res.faithfulness_score,
                "target_hit": hit,
                "retrieved_ids": [s.get("id") for s in res.sources],
                "citations": [
                    {"video": c.video_id, "time": f"{c.timestamp_start}-{c.timestamp_end}", "faithful": c.is_faithful}
                    for c in res.citations
                ]
            })
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            
    avg_latency = total_latency / total_queries if total_queries > 0 else 0
    avg_recall = total_recall / total_queries if total_queries > 0 else 0
    avg_faithfulness = total_faithfulness / total_queries if total_queries > 0 else 0
    
    print("\n=== TOPICAL EVALUATION REPORT ===")
    print(f"Total Queries: {total_queries}")
    print(f"Average Recall@3: {avg_recall:.2%}")
    print(f"Average Faithfulness: {avg_faithfulness:.2%}")
    print(f"Average Latency: {avg_latency:.2f} seconds")
    
    with open("evaluation_results_topical.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to evaluation_results_topical.json")

if __name__ == "__main__":
    run_evaluation()
