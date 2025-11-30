import os

from dotenv import load_dotenv
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
# STEP 1 — FETCH RAW TRANSCRIPTS
# -------------------------

def get_raw_transcripts(
    supabase: Client,
    video_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:

    print("\n=== STEP 1: Fetching Raw Transcripts ===")

    query = supabase.table("raw_transcripts").select("*")

    if video_ids:
        print(f"Fetching transcripts for video IDs: {video_ids}")
        query = query.in_("video_id", video_ids)
    else:
        print("Fetching ALL transcripts...")

    result = query.execute()

    data = result.data if hasattr(result, "data") else result

    print(f"Fetched {len(data)} transcript(s).")
    return data


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
# STEP 2 — CLEAN + SPEAKER IDENTIFICATION
# -------------------------

def clean_and_identify_speakers(client: OpenAI, raw_transcript: str) -> str:

    pretty_print("RAW TRANSCRIPT (BEFORE PROCESSING)", raw_transcript)

    prompt = f"""
You are processing a raw, messy dialogue transcript from a Muslim Lantern dawah video. 
Your task is to output a clean, readable conversation with accurate speaker turns.

RULES:
- Only two speakers:
    Muslim Lantern
    Non-Muslim
- Fix transcription errors, incomplete words, and repeated fragments.
- Preserve the meaning and order of the dialogue.
- Do NOT invent new arguments or add new information.
- If a speaker is unclear, infer the most likely speaker based on context.
- Remove filler words like “uh”, “umm”, “you know” unless meaningful.
- Do NOT shorten or summarize the content — keep the full conversation.
- Keep responses in natural speaking style.

OUTPUT FORMAT (MUST FOLLOW EXACTLY):

Muslim Lantern: ...
Non-Muslim: ...
Muslim Lantern: ...
Non-Muslim: ...
(continue)

RAW TRANSCRIPT:
{raw_transcript}
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )

    cleaned = response.output_text

    pretty_print("CLEANED + SPEAKER IDENTIFIED TRANSCRIPT (AFTER GPT)", cleaned)

    return cleaned


# -------------------------
# STEP 3 — STORE FORMATTED TRANSCRIPT
# -------------------------

def store_formatted_transcript(
    supabase: Client,
    video_id: str,
    formatted_transcript: str
):
    print(f"\n=== STEP 3: Saving cleaned transcript for video: {video_id} ===")

    data = {
        "video_id": video_id,
        "formatted_transcript": formatted_transcript
    }

    # Use UPSERT so we can safely re-run the pipeline
    result = (
        supabase
        .table("formatted_transcripts")
        .upsert(data)  # on_conflict defaults to the PK (video_id)
        .execute()
    )

    if getattr(result, "error", None):
        print("Error saving:", result.error)
    else:
        print(f"Successfully saved formatted transcript for {video_id}")



# -------------------------
# FULL PIPELINE
# -------------------------

if __name__ == "__main__":

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "OPENAI-API-KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "PINECONE-API-KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    supabase = init_supabase(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    openai_client = init_openai(OPENAI_API_KEY)

    # Fetch raw transcripts
    raw_items = get_raw_transcripts(supabase)

    # Process each transcript
    for item in raw_items:
        video_id = item["video_id"]
        raw_text = item["transcript"]

        print(f"\n\n============================")
        print(f"PROCESSING VIDEO: {video_id}")
        print("============================")

        # Step 2 — Clean with GPT
        cleaned = clean_and_identify_speakers(openai_client, raw_text)

        # Step 3 — Save to Supabase
        store_formatted_transcript(supabase, video_id, cleaned)

    print("\nALL TRANSCRIPTS PROCESSED!\n")
