import time
import json
from typing import List, Dict
from rag_core import RAGSystem

import time
import json
import os
from typing import List, Dict
from rag_core import RAGSystem

GENERATED_EVAL_FILE = "generated_eval_set.json"

def run_evaluation():
    print("Initializing RAG System...")
    rag = RAGSystem()
    
    # Load generated queries
    if not os.path.exists(GENERATED_EVAL_FILE):
        print(f"Error: {GENERATED_EVAL_FILE} not found. Run generate_eval_dataset.py first.")
        return

    with open(GENERATED_EVAL_FILE, "r") as f:
        eval_data = json.load(f)
    
    total_faithfulness = 0.0
    total_latency = 0.0
    total_recall = 0.0
    total_queries = len(eval_data)
    
    results = []
    
    print(f"\nStarting evaluation on {total_queries} queries from {GENERATED_EVAL_FILE}...\n")
    
    for i, item in enumerate(eval_data):
        query = item["query"]
        target_ids = set(item.get("relevant_supabase_ids", []))
        
        print(f"[{i+1}/{total_queries}] Query: {query}")
        
        try:
            res = rag.query(query)
            
            total_faithfulness += res.faithfulness_score
            total_latency += res.latency_seconds
            
            # Check Recall: Did we retrieve the chunk that generated the question?
            # rag.retrieve() returns context items with "indices" (Supabase IDs)
            retrieved_ids = set()
            for ctx in res.context_used:
                # In rag_core.retrieve, we added "indices" to the context item dict?
                # Wait, I need to check if I exposed "indices" in the return value of retrieve.
                # In rag_core.py:
                # context_items.append({ ..., "indices": indices }) ??
                # Looking at my previous edit to rag_core.py:
                # I did NOT add "indices" to the final context_items list.
                # I added "pairs": pairs_data.
                # And pairs_data has "pair_id".
                # I need to map pair_id back to Supabase ID or just check if the content matches.
                
                # Actually, I can infer the ID from pair_id if I trust the mapping, 
                # OR I can modify rag_core to expose the Supabase ID.
                # Let's modify rag_core to expose it, OR just rely on Faithfulness for now.
                
                # But the user complained about "poor results", implying retrieval might be bad.
                # Let's check if the *answer* cites the correct ID.
                # The citation object has pair_id.
                
                # Let's try to match pair_id to target_ids.
                # target_ids are integers (Supabase IDs).
                # pair_id is string "video_id_index".
                # We need to know the mapping.
                
                # In rag_core.py, I did:
                # idx_str = pid.split("_")[-1]
                # indices.append(int(idx_str))
                # So I can do the same here.
                pass

            # Calculate Recall based on citations
            # If any cited pair maps to a target ID, we count it as a hit?
            # Or better: Did the *Retrieval* step find it?
            
            # Let's calculate "Answer Recall": Does the answer cite the relevant source?
            hit = False
            for cit in res.citations:
                try:
                    idx = int(cit.pair_id.split("_")[-1])
                    if idx in target_ids:
                        hit = True
                        break
                except:
                    pass
            
            if hit:
                total_recall += 1.0
            
            print(f"  -> Latency: {res.latency_seconds:.2f}s")
            print(f"  -> Faithfulness: {res.faithfulness_score:.2f}")
            print(f"  -> Target Hit: {hit}")
            
            results.append({
                "query": query,
                "answer": res.answer,
                "faithfulness": res.faithfulness_score,
                "latency": res.latency_seconds,
                "target_hit": hit,
                "citations": [
                    {"id": f"{c.video_id}:{c.pair_id}", "faithful": c.is_faithful} 
                    for c in res.citations
                ]
            })
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            
    avg_faithfulness = total_faithfulness / total_queries if total_queries > 0 else 0
    avg_latency = total_latency / total_queries if total_queries > 0 else 0
    avg_recall = total_recall / total_queries if total_queries > 0 else 0
    
    print("\n=== EVALUATION REPORT ===")
    print(f"Total Queries: {total_queries}")
    print(f"Average Faithfulness: {avg_faithfulness:.2%}")
    print(f"Average Answer Recall: {avg_recall:.2%}")
    print(f"Average Latency: {avg_latency:.2f} seconds")
    
    with open("evaluation_results_v2_generated.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to evaluation_results_v2_generated.json")

if __name__ == "__main__":
    run_evaluation()
