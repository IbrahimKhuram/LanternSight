# pipeline/timestamps.py

import os
import re
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL is missing.")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY is missing.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)


# ====================================================
# STEP 1 — Fetch formatted transcript
# ====================================================

def fetch_formatted_transcript(video_id: str) -> str:
    """Fetch formatted transcript from Supabase."""
    response = (
        supabase.table("formatted_transcripts")
        .select("formatted_text")
        .eq("video_id", video_id)
        .single()
        .execute()
    )

    if not response.data:
        raise Exception(f"No formatted transcript found for {video_id}")

    return response.data["formatted_text"]


# ====================================================
# STEP 2 — Fetch topics for this video_id
# ====================================================

def fetch_topical_answers(video_id: str):
    """
    Returns list of rows from topical_answers table:
    Each row contains: id, topic, answer
    """
    response = (
        supabase.table("topical_answers")
        .select("*")
        .eq("video_id", video_id)
        .execute()
    )

    if not response.data:
        raise Exception(f"No topical answers found for {video_id}")

    return response.data


# ====================================================
# STEP 3 — GPT timestamp extraction
# ====================================================

def generate_timestamps(formatted_transcript: str, topical_answers: list) -> str:
    """
    Ask GPT-4.1 to assign timestamp_start and timestamp_end
    to each topic based on formatted transcript.
    """

    topic_list_str = ""
    for i, t in enumerate(topical_answers, start=1):
        topic_list_str += f"topic {i}: {t['topic']}\n"

    SYSTEM_PROMPT = """
You assign timestamps for each topic based on the formatted transcript.

RULES:
- Compare the content of each topic with the transcript.
- Identify where the topic begins and ends based on spoken lines.
- Return timestamps in EXACT format:

topic 1:
timestamp_start: 00:00:00
timestamp_end: 00:00:00

topic 2:
timestamp_start: 00:00:00
timestamp_end: 00:00:00

- If unsure, give best estimate.
- Do NOT invent transcript lines.
- Timestamps must be in HH:MM:SS format.
"""

    USER_MESSAGE = f"""
Formatted Transcript:
{formatted_transcript}

Topics:
{topic_list_str}
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_MESSAGE},
        ],
        temperature=0
    )

    return response.output_text.strip()


# ====================================================
# STEP 4 — Parse GPT output + store timestamps
# ====================================================

def parse_and_save_timestamps(video_id: str, topics: list, timestamps_text: str):
    """
    Parse GPT timestamp output and store timestamp_start + timestamp_end
    into Supabase topical_answers_test table.
    """

    pairs = re.split(r"topic\s*:?\s*\d+\s*:?", timestamps_text, flags=re.IGNORECASE)
    entries = [p.strip() for p in pairs if p.strip()]

    if len(entries) != len(topics):
        print("⚠ WARNING: Topic count mismatch. Attempting best-effort matching.")

    for idx, entry in enumerate(entries):
        ts_start = re.search(r"timestamp_start:\s*([0-9:]+)", entry)
        ts_end = re.search(r"timestamp_end:\s*([0-9:]+)", entry)

        start_val = ts_start.group(1) if ts_start else None
        end_val = ts_end.group(1) if ts_end else None

        topic_row = topics[idx]

        update_data = {
            "timestamp_start": start_val,
            "timestamp_end": end_val
        }

        supabase.table("topical_answers_test") \
            .update(update_data) \
            .eq("id", topic_row["id"]) \
            .execute()

    return True