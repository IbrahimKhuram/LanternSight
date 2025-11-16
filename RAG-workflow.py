import os
import json
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# --------------------------------------
# CONFIGURATION
# --------------------------------------
OPENAI_API_KEY = "OPENAI-API-KEY"
PINECONE_API_KEY = "PINECONE-API-KEY"
TRANSCRIPT_DIR = r"D:\transcript_dataset"

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

# Ensure index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Get index handle
index = pc.Index(INDEX_NAME)

# --------------------------------------
# Step 1 - Extract Claim–Response Pairs
# --------------------------------------
def extract_pairs(transcript_text):
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
            {"role": "user", "content": transcript_text}
        ],
        temperature=0
    )

    text = result.choices[0].message.content
    return json.loads(text)

# --------------------------------------
# Step 2 - Token-based Chunking
# --------------------------------------
def chunk_pairs(pairs, video_id):
    chunks = []
    cur_tokens = []
    cur_meta = []

    def flush():
        if not cur_tokens:
            return
        text = encoder.decode(cur_tokens)
        chunks.append({
            "text": text,
            "metadata": {
                "video_id": video_id,
                "pairs": cur_meta.copy()
            }
        })
        cur_tokens.clear()
        cur_meta.clear()

    for p in pairs:
        block = json.dumps(p, ensure_ascii=False)
        block_tokens = encoder.encode(block)

        if len(cur_tokens) + len(block_tokens) > MAX_TOKENS:
            flush()

        cur_tokens.extend(block_tokens)
        cur_meta.append({
            "claim": p["claim"],
            "start": p.get("start", 0),
            "end": p.get("end", 0)
        })

    flush()
    return chunks

# --------------------------------------
# Step 3 - Embedding helper
# --------------------------------------
def embed(text: str):
    result = client.embeddings.create(model=EMBED_MODEL, input=text)
    return result.data[0].embedding

# --------------------------------------
# Step 4 - Upsert chunks into Pinecone
# --------------------------------------
def index_chunks(video_id, chunks):
    vectors = []
    for i, ch in enumerate(chunks):
        emb = embed(ch["text"])
        vec_id = f"{video_id}-chunk-{i}"
        vectors.append({
            "id": vec_id,
            "values": emb,
            "metadata": ch["metadata"]
        })
    index.upsert(vectors=vectors)
    print(f"Indexed: {video_id} → {len(vectors)} chunks")

# --------------------------------------
# Step 5 - RAG Query
# --------------------------------------
def rag_query(query: str, top_k=5):
    q_emb = embed(query)
    results = index.query(vector=q_emb, top_k=top_k, include_metadata=True)

    retrieved_context = json.dumps(results.matches, indent=2)

    prompt = f"""
    Answer the user's query using ONLY the retrieved context:

    Query: {query}

    Retrieved context:
    {retrieved_context}

    Include timestamps and video_id in your answer.
    """

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = resp.choices[0].message.content
    return answer, results.matches

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

        pairs = extract_pairs(transcript_text)
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
    print("REFERENCES:\n", refs)
