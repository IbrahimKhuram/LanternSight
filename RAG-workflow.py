import os
import json
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# this is just GPT'd i will fix it later
# --------------------------------------
# CONFIGURATION
# --------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "OPENAI-API-KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "PINECONE-API-KEY")

TRANSCRIPT_DIR = "transcripts"

INDEX_NAME = "claims-index"
GPT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
MAX_TOKENS = 300  # token-based chunking

# --------------------------------------
# CLIENTS
# --------------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
encoder = tiktoken.get_encoding("cl100k_base")

# Create Pinecone client instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# --------------------------------------
# Ensure index exists
# IMPORTANT: embedding dim for text-embedding-3-small is 1536 by default
# so Pinecone index must also be 1536-dimensional.
# --------------------------------------
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        vector_type="dense",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Get index handle
index = pc.Index(INDEX_NAME)


# --------------------------------------
# Step 1 - Extract Claim–Response Pairs
# --------------------------------------
def extract_pairs(transcript_text: str):
    system_prompt = """
    Extract CLAIM–RESPONSE pairs from the transcript.

    Return ONLY a JSON array:
    [
      {
        "claim": "...",
        "response": "...",
        "start": 0,
        "end": 10
      }
    ]

    If timestamps are missing, estimate start/end roughly by order.
    """

    result = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript_text},
        ],
        temperature=0,
    )

    text = result.choices[0].message.content.strip()

    # --- Make JSON parsing a bit more robust (handles ```json fences etc.) ---
    # Grab the first JSON array in the response.
    try:
        # Try direct parse first
        return json.loads(text)
    except json.JSONDecodeError:
        # Very simple "extract JSON array" fallback
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            json_str = text[start: end + 1]
            return json.loads(json_str)
        raise


# --------------------------------------
# Step 2 - Token-based Chunking
#   We pack multiple pairs into chunks of ~MAX_TOKENS and
#   preserve both claim and response in metadata for RAG.
# --------------------------------------
def chunk_pairs(pairs, video_id: str):
    chunks = []
    cur_tokens = []
    cur_pairs = []

    def flush():
        if not cur_tokens:
            return
        text = encoder.decode(cur_tokens)
        # Store text also in metadata so we can show it at query time
        chunks.append(
            {
                "text": text,
                "metadata": {
                    "video_id": video_id,
                    "pairs": cur_pairs.copy(),
                },
            }
        )
        cur_tokens.clear()
        cur_pairs.clear()

    for p in pairs:
        # keep full pair information
        cur_pairs.append(
            {
                "claim": p.get("claim", ""),
                "response": p.get("response", ""),
                "start": p.get("start", 0),
                "end": p.get("end", 0),
            }
        )
        block = json.dumps(cur_pairs[-1], ensure_ascii=False)
        block_tokens = encoder.encode(block)

        # if adding this block would exceed limit, flush current chunk first
        if len(cur_tokens) + len(block_tokens) > MAX_TOKENS:
            flush()

        cur_tokens.extend(block_tokens)

    flush()
    return chunks


# --------------------------------------
# Step 3 - Embedding helper
# --------------------------------------
def embed(text: str):
    result = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return result.data[0].embedding


# --------------------------------------
# Step 4 - Upsert chunks into Pinecone
# --------------------------------------
def index_chunks(video_id: str, chunks):
    vectors = []
    for i, ch in enumerate(chunks):
        emb = embed(ch["text"])
        vec_id = f"{video_id}-chunk-{i}"

        # Extract metadata from chunk
        video_id_meta = ch["metadata"]["video_id"]
        pairs = ch["metadata"]["pairs"]  # list[dict]

        # Pinecone metadata must be simple types – store pairs as JSON string
        metadata = {
            "video_id": video_id_meta,
            "text": ch["text"],
            "pairs_json": json.dumps(pairs, ensure_ascii=False),
        }

        vectors.append(
            {
                "id": vec_id,
                "values": emb,
                "metadata": metadata,
            }
        )

    if vectors:
        index.upsert(vectors=vectors)
        print(f"Indexed: {video_id} → {len(vectors)} chunks")
    else:
        print(f"Nothing to index for {video_id}")


# --------------------------------------
# Step 5 - RAG Query
# --------------------------------------
def rag_query(query: str, top_k: int = 5):
    q_emb = embed(query)
    results = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True,
    )

    context_items = []
    for m in results.matches:
        md = m.metadata or {}
        pairs_json = md.get("pairs_json", "[]")
        try:
            pairs = json.loads(pairs_json)
        except json.JSONDecodeError:
            pairs = []

        context_items.append(
            {
                "score": m.score,
                "video_id": md.get("video_id"),
                "text": md.get("text"),
                "pairs": pairs,
            }
        )

    retrieved_context = json.dumps(
        context_items,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are a QA assistant over claim–response pairs from video transcripts.

Answer the user's query using ONLY the retrieved context below.
If the context is insufficient, say you don't know.

Query:
{query}

Retrieved context (JSON list of chunks, with video_id, text, and pairs):
{retrieved_context}

When answering:
- Quote / paraphrase the relevant claim–response content.
- Include video_id and any available start/end timestamps in your answer.
"""

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    answer = resp.choices[0].message.content
    return answer, context_items


# --------------------------------------
# Step 6 - Process all transcripts
# --------------------------------------
def process_all_transcripts():
    files = [f for f in os.listdir(TRANSCRIPT_DIR) if f.endswith(".txt")]
    for filename in files:
        video_id = filename.replace(".txt", "")
        path = os.path.join(TRANSCRIPT_DIR, filename)

        print(f"\n=== Processing {filename} ===")
        with open(path, "r", encoding="utf-8") as f:
            transcript_text = f.read()

        if not transcript_text.strip():
            print("  -> Empty transcript, skipping.")
            continue

        pairs = extract_pairs(transcript_text)
        if not pairs:
            print("  -> No pairs extracted, skipping.")
            continue

        chunks = chunk_pairs(pairs, video_id)
        index_chunks(video_id, chunks)


# --------------------------------------
# RUN
# --------------------------------------
if __name__ == "__main__":
    process_all_transcripts()

    print("\n--- SAMPLE QUERY ---")
    ans, refs = rag_query("What did the Muslim speaker say about Hindu scriptures?")
    print("ANSWER:\n", ans)
    print("\nREFERENCES:\n", json.dumps(refs, ensure_ascii=False, indent=2))
