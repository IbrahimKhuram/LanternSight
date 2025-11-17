from supabase import create_client, Client
from typing import Optional, List, Dict, Any
from openai import OpenAI
import textwrap

# -------------------------
# INITIALIZATION
# -------------------------


def init_supabase(url: str, key: str) -> Client:
    return create_client(url, key)

def init_openai(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

# -------------------------
# Helper: pretty printing
# -------------------------

def pretty_print(title: str, content: str, max_chars: int = 900):
    print(f"\n----- {title} -----")
    if len(content) > max_chars:
        print(textwrap.fill(content[:max_chars], width=90))
        print("\n...[OUTPUT TRUNCATED]...\n")
    else:
        print(textwrap.fill(content, width=90))


# -------------------------
# STEP 4 — FETCH FORMATTED TRANSCRIPTS
# -------------------------

def get_formatted_transcripts(
    supabase: Client,
    video_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:

    print("\n=== STEP 4: Fetching Formatted Transcripts ===")

    query = supabase.table("formatted_transcripts").select("*")

    if video_ids:
        print(f"Fetching formatted transcripts for video IDs: {video_ids}")
        query = query.in_("video_id", video_ids)
    else:
        print("Fetching ALL formatted transcripts...")

    result = query.execute()
    data = result.data if hasattr(result, "data") else result

    print(f"Fetched {len(data)} formatted transcript(s).")
    return data

# -------------------------
# STEP 5 — EXTRACT CLAIM-RESPONSE PAIRS
# -------------------------

def extract_claim_response_pairs(client: OpenAI, formatted_transcript: str) -> List[Dict[str, str]]:
    """
    Extract Non-Muslim claims and Muslim Lantern responses from transcript.
    Returns a list of dicts: {'claim': ..., 'response': ...}
    """

    prompt = f"""
You are processing a cleaned, speaker-identified transcript from a Muslim Lantern dawah video.

RULES:
- Extract every statement by Non-Muslim that is a claim, question, or objection.
- Pair it with the **immediate response** by Muslim Lantern.
- Keep order intact.
- Do NOT invent content or summarize; preserve meaning.

OUTPUT FORMAT (MUST FOLLOW EXACTLY):

"claim": "Non-Muslim statement here",
"response": "Muslim Lantern response here"
...

Transcript:
{formatted_transcript}
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )

    import json
    output_text = response.output_text

    try:
        pairs = json.loads(output_text)
        if isinstance(pairs, list):
            return pairs
        else:
            print("⚠️ Warning: GPT output not a list, returning empty list")
            return []
    except json.JSONDecodeError:
        print("⚠️ Warning: Could not parse GPT output as JSON")
        return []

# -------------------------
# STEP 6 — STORE CLAIM-RESPONSE PAIRS
# -------------------------

def store_claim_response_pairs(
    supabase: Client,
    video_id: str,
    pairs: List[Dict[str, str]]
):
    print(f"\n=== STEP 6: Saving claim-response pairs for video: {video_id} ===")

    for idx, pair in enumerate(pairs, start=1):
        row_id = f"{video_id}_{idx}"
        data = {
            "id": row_id,
            "claim": pair.get("claim", ""),
            "response": pair.get("response", "")
        }
        result = supabase.table("claim_response_pairs").insert(data).execute()

        if getattr(result, "error", None):
            print(f"❌ Error saving pair {row_id}: {result.error}")
        else:
            print(f"✅ Saved pair {row_id}")

# -------------------------
# FULL PIPELINE INCLUDING CLAIM-RESPONSE EXTRACTION
# -------------------------

if __name__ == "__main__":

    SUPABASE_URL=
    SUPABASE_SERVICE_ROLE_KEY=
    OPENAI_API_KEY=

    supabase = init_supabase(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    openai_client = init_openai(OPENAI_API_KEY)

    # Step 4: fetch formatted transcripts
    formatted_items = get_formatted_transcripts(supabase)

    # Step 5 + 6: extract claim-response pairs & store
    for item in formatted_items:
        video_id = item["video_id"]
        formatted_text = item["formatted_transcript"]

        pairs = extract_claim_response_pairs(openai_client, formatted_text)
        if pairs:
            pretty_print(f"Extracted {len(pairs)} claim-response pairs for {video_id}", str(pairs)[:900])
        store_claim_response_pairs(supabase, video_id, pairs)

    print("\n🎉 ALL TRANSCRIPTS AND CLAIM-RESPONSE PAIRS PROCESSED SUCCESSFULLY!\n")
