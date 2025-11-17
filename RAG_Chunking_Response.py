import os
import json
import math
from typing import List, Dict, Any, Optional, Tuple

import tiktoken
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


# -------------------------
# INITIALIZATION
# -------------------------

INDEX_NAME = "claims-index"
EMBED_MODEL = "text-embedding-3-small"
GPT_EXTRACTION_MODEL = "gpt-4.1-mini"
GPT_RAG_MODEL = "gpt-4.1-mini"
MAX_TOKENS = 300  # chunk size for claims/responses


def init_supabase(url: str, key: str) -> Client:
    return create_client(url, key)


def init_openai(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def init_pinecone(api_key: str) -> Tuple[Pinecone, Any]:
    pc = Pinecone(api_key=api_key)
    if not pc.has_index(INDEX_NAME):
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            vector_type="dense",
            dimension=1536,  # text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(INDEX_NAME)
    return pc, index


encoder = tiktoken.get_encoding("cl100k_base")


# -------------------------
# STEP 4 — FETCH CLEANED TRANSCRIPTS
# -------------------------

def get_formatted_transcripts(
    supabase: Client,
    video_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    print("\n=== STEP 4: Fetching formatted transcripts ===")

    query = supabase.table("formatted_transcripts").select("*")
    if video_ids:
        query = query.in_("video_id", video_ids)

    result = query.execute()
    data = result.data if hasattr(result, "data") else result
    print(f"Fetched {len(data)} formatted transcript(s).")
    return data


# -------------------------
# STEP 5 — EXTRACT CLAIM–RESPONSE PAIRS
# -------------------------

def extract_claim_response_pairs(
    client: OpenAI,
    formatted_transcript: str,
    video_id: str,
) -> List[Dict[str, Any]]:
    """
    Use GPT to turn a cleaned, speaker-labelled transcript into
    structured claim–response pairs. Each pair gets a stable pair_id:
    <video_id>_<index>.
    """
    prompt = f"""
You are given a clean dialogue transcript between two speakers:

- Muslim Lantern
- Non-Muslim

Your task is to extract CLAIM–RESPONSE pairs, where:
- A "claim" is a clear statement or objection raised by either speaker.
- A "response" is the direct reply to that claim.

Return ONLY valid JSON with this exact structure:

[
  {{
    "pair_id": "VIDEOID_001",
    "claim_speaker": "Non-Muslim",
    "claim": "...",
    "response_speaker": "Muslim Lantern",
    "response": "...",
    "start": 0,
    "end": 10
  }},
  ...
]

Rules:
- Use pair_id = "<video_id>_XXX" with 3-digit numbering starting from 001.
- Do NOT summarise; keep the main wording of each claim/response.
- Use best-effort estimates for start/end seconds if timestamps are not known.
- Speakers must be exactly "Muslim Lantern" or "Non-Muslim".

Transcript:
{formatted_transcript}
""".strip()

    resp = client.responses.create(
        model=GPT_EXTRACTION_MODEL,
        input=prompt,
    )
    text = resp.output_text.strip()

    # robust JSON parsing
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    # ensure pair_ids exist / are normalized
    cleaned_pairs = []
    for i, p in enumerate(data, start=1):
        pair_id = p.get("pair_id") or f"{video_id}_{i:03d}"
        cleaned_pairs.append(
            {
                "pair_id": pair_id,
                "video_id": video_id,
                "claim_speaker": p.get("claim_speaker", ""),
                "claim": p.get("claim", ""),
                "response_speaker": p.get("response_speaker", ""),
                "response": p.get("response", ""),
                "start": p.get("start", 0),
                "end": p.get("end", 0),
            }
        )

    print(f"Extracted {len(cleaned_pairs)} pair(s) for video {video_id}")
    return cleaned_pairs


# -------------------------
# STEP 6 — STORE PAIRS IN SUPABASE
# -------------------------

def store_claim_response_pairs(
    supabase: Client, pairs: List[Dict[str, Any]]
):
    if not pairs:
        return

    print(f"Saving {len(pairs)} claim–response pair(s) to Supabase...")

    rows = []
    for p in pairs:
        row = {
            "id": p["pair_id"],       # PK: VIDEOID_XXX
            "claim": p["claim"],
            "response": p["response"],
        }
        rows.append(row)

    # upsert avoids duplicate-key errors when re-running the pipeline
    result = (
        supabase
        .table("claim_response_pairs")
        .upsert(rows)
        .execute()
    )

    if getattr(result, "error", None):
        print("Error saving pairs:", result.error)
    else:
        print("Pairs saved successfully.")




# -------------------------
# STEP 7 — CHUNKING FOR RETRIEVAL
# -------------------------

def chunk_pairs_for_index(
    pairs: List[Dict[str, Any]],
    max_tokens: int = MAX_TOKENS,
) -> List[Dict[str, Any]]:
    """
    Turn a list of pairs (for a single video) into tiktoken-bounded chunks.
    Each chunk contains one or more pairs and carries metadata:
    - video_id
    - pair_ids[]
    - pairs_json (stringified list of pairs)
    """
    if not pairs:
        return []

    video_id = pairs[0]["video_id"]
    chunks: List[Dict[str, Any]] = []
    cur_tokens: List[int] = []
    cur_pairs: List[Dict[str, Any]] = []
    chunk_idx = 0

    def flush():
        nonlocal chunk_idx
        if not cur_tokens:
            return
        chunk_text = encoder.decode(cur_tokens)
        metadata = {
            "video_id": video_id,
            "pair_ids": [p["pair_id"] for p in cur_pairs],
            "pairs_json": json.dumps(cur_pairs, ensure_ascii=False),
        }
        chunks.append(
            {
                "id": f"{video_id}-chunk-{chunk_idx}",
                "text": chunk_text,
                "metadata": metadata,
            }
        )
        chunk_idx += 1
        cur_tokens.clear()
        cur_pairs.clear()

    for p in pairs:
        pair_payload = {
            "pair_id": p["pair_id"],
            "claim_speaker": p["claim_speaker"],
            "claim": p["claim"],
            "response_speaker": p["response_speaker"],
            "response": p["response"],
            "start": p["start"],
            "end": p["end"],
        }
        block = json.dumps(pair_payload, ensure_ascii=False)
        block_tokens = encoder.encode(block)

        if len(cur_tokens) + len(block_tokens) > max_tokens:
            flush()

        cur_pairs.append(pair_payload)
        cur_tokens.extend(block_tokens)

    flush()
    print(f"Created {len(chunks)} chunk(s) for video {video_id}")
    return chunks


# -------------------------
# EMBEDDINGS & INDEXING
# -------------------------

def embed_text(client: OpenAI, text: str) -> List[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def upsert_chunks_to_pinecone(
    index: Any,
    client: OpenAI,
    chunks: List[Dict[str, Any]],
):
    if not chunks:
        return

    vectors = []
    for ch in chunks:
        emb = embed_text(client, ch["text"])
        md = ch["metadata"]
        vectors.append(
            {
                "id": ch["id"],
                "values": emb,
                "metadata": {
                    "video_id": md["video_id"],
                    "pair_ids": md["pair_ids"],
                    "pairs_json": md["pairs_json"],
                },
            }
        )

    index.upsert(vectors=vectors)
    print(f"Indexed {len(vectors)} chunk(s) into Pinecone.")


# -------------------------
# RAG QUERY
# -------------------------

def rag_query(
    client: OpenAI,
    index: Any,
    query: str,
    top_k: int = 5,
) -> Tuple[str, List[Dict[str, Any]]]:

    q_emb = embed_text(client, query)
    results = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True,
    )

    context_items: List[Dict[str, Any]] = []
    for m in results.matches:
        md = m.metadata or {}
        pair_ids = md.get("pair_ids", [])
        try:
            pairs = json.loads(md.get("pairs_json", "[]"))
        except json.JSONDecodeError:
            pairs = []

        context_items.append(
            {
                "score": m.score,
                "video_id": md.get("video_id"),
                "pair_ids": pair_ids,
                "pairs": pairs,
            }
        )

    rag_context = json.dumps(
        context_items,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are a QA assistant over claim–response pairs from dawah videos.

Use ONLY the context below to answer the question. If the context
is insufficient, say you don't know.

Question:
{query}

Context (JSON chunks with video_id, pair_ids, and pairs):
{rag_context}

In your answer:
- Use the content of the claims and responses.
- Mention video_id and pair_id(s) where relevant.
""".strip()

    resp = client.responses.create(
        model=GPT_RAG_MODEL,
        input=prompt,
    )
    answer = resp.output_text.strip()
    return answer, context_items


# -------------------------
# METRICS: Recall@k and nDCG@k
# -------------------------

def dcg(relevances: List[int]) -> float:
    return sum(
        rel / math.log2(idx + 2)  # idx starts at 0
        for idx, rel in enumerate(relevances)
    )


def evaluate_retriever(
    client: OpenAI,
    index: Any,
    dev_queries: List[Dict[str, Any]],
    k_values: List[int] = [3, 5, 10],
):
    print("\n=== EVALUATION: Recall@k and nDCG@k ===")

    # metrics[q][k] = (recall, ndcg)
    metrics: Dict[int, Dict[int, Tuple[float, float]]] = {}

    for qi, item in enumerate(dev_queries):
        query = item["query"]
        relevant = set(item["relevant_pair_ids"])
        if not relevant:
            continue

        max_k = max(k_values)
        q_emb = embed_text(client, query)
        res = index.query(
            vector=q_emb,
            top_k=max_k,
            include_metadata=True,
        )

        # Flatten pair_ids in retrieval order (per chunk).
        ranked_pairs: List[str] = []
        for match in res.matches:
            md = match.metadata or {}
            for pid in md.get("pair_ids", []):
                if pid not in ranked_pairs:
                    ranked_pairs.append(pid)

        metrics[qi] = {}
        for k in k_values:
            top_k_pairs = ranked_pairs[:k]
            # Recall@k
            retrieved_set = set(top_k_pairs)
            intersection = len(retrieved_set & relevant)
            recall_k = intersection / len(relevant)

            # nDCG@k
            relevances = [1 if pid in relevant else 0 for pid in top_k_pairs]
            dcg_k = dcg(relevances)
            ideal_rels = [1] * min(len(relevant), k)
            idcg_k = dcg(ideal_rels) or 1.0
            ndcg_k = dcg_k / idcg_k

            metrics[qi][k] = (recall_k, ndcg_k)

    # Aggregate across queries
    for k in k_values:
        recalls = []
        ndcgs = []
        for qi in metrics:
            r, n = metrics[qi][k]
            recalls.append(r)
            ndcgs.append(n)
        if not recalls:
            continue
        print(
            f"k={k}: "
            f"Recall@{k}={sum(recalls)/len(recalls):.3f}, "
            f"nDCG@{k}={sum(ndcgs)/len(ndcgs):.3f}"
        )


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not all([OPENAI_API_KEY, PINECONE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY]):
        raise RuntimeError("Missing one or more required environment variables.")

    supabase = init_supabase(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    openai_client = init_openai(OPENAI_API_KEY)
    pc, pinecone_index = init_pinecone(PINECONE_API_KEY)

    # ----- Ingestion: formatted transcripts -> pairs -> chunks -> Pinecone -----

    formatted_items = get_formatted_transcripts(supabase)

    for item in formatted_items:
        video_id = item["video_id"]
        formatted_text = item["formatted_transcript"]

        print("\n============================")
        print(f"PROCESSING VIDEO FOR RAG: {video_id}")
        print("============================")

        # Step 5: extract pairs
        pairs = extract_claim_response_pairs(openai_client, formatted_text, video_id)

        # Step 6: store pairs in Supabase
        store_claim_response_pairs(supabase, pairs)

        # Step 7: chunk and index
        chunks = chunk_pairs_for_index(pairs)
        upsert_chunks_to_pinecone(pinecone_index, openai_client, chunks)

    # ----- Example RAG query -----
    example_query = "What did the Muslim speaker say about Hindu scriptures?"
    answer, ctx = rag_query(openai_client, pinecone_index, example_query, top_k=5)
    print("\n=== SAMPLE RAG ANSWER ===")
    print(answer)
    print("\n=== CONTEXT USED ===")
    print(json.dumps(ctx, ensure_ascii=False, indent=2))

    # ----- Example dev set & metrics -----
    DEV_QUERIES = [
        {
            "query": "How did the Muslim respond to the claim that all religions are the same?",
            "relevant_pair_ids": ["VIDEO1_001", "VIDEO1_002"],
        },
        {
            "query": "What objection about the Trinity was raised?",
            "relevant_pair_ids": ["VIDEO2_003"],
        },
    ]
    evaluate_retriever(openai_client, pinecone_index, DEV_QUERIES, k_values=[3, 5])
