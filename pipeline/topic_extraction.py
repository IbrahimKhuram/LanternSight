# pipeline/topic_extraction.py

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
    raise ValueError("SUPABASE_URL is missing — check your .env")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY is missing.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)


# -----------------------------------------------------
# Fetch Clean Transcript
# -----------------------------------------------------

def fetch_clean_transcript(video_id: str) -> str:
    response = (
        supabase.table("formatted_transcripts")
        .select("clean_text")
        .eq("video_id", video_id)
        .single()
        .execute()
    )

    if not response.data:
        raise Exception(f"No cleaned transcript found for video_id: {video_id}")

    return response.data["clean_text"]


# -----------------------------------------------------
# GPT STAGE 1 — Extract Non-Muslim Objections + Timestamps
# -----------------------------------------------------

def extract_objections(clean_text: str) -> str:

    SYSTEM_MESSAGE = """
Extract a list of Non-Muslim objections that Muslim Lantern refuted from a Muslim Lantern dawah conversation.

Go through the video transcript and:
- Identify the Non-Muslim objection.
- Identify timestamp_start = when the objection begins.
- Identify timestamp_end = when Muslim Lantern COMPLETES refuting that objection.


Output format:

topic 1:
objection: <non-muslim objection>
timestamp_start: <HH:MM:SS or MM:SS>
timestamp_end: <HH:MM:SS or MM:SS>

topic 2:
objection: <non-muslim objection>
timestamp_start: <HH:MM:SS or MM:SS>
timestamp_end: <HH:MM:SS or MM:SS>

(continue)

"""

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": clean_text},
        ],
        temperature=0,
    )

    return response.output_text.strip()


# -----------------------------------------------------
# Parse stage-1 output
# -----------------------------------------------------

def parse_stage1(raw: str):
    """
    Returns a list of:
    [
        {
            "topic_num": 1,
            "objection": "...",
            "timestamp_start": "...",
            "timestamp_end": "..."
        },
        ...
    ]
    """
    topics = []

    # Case-insensitive split on 'topic '
    blocks = re.split(r'(?i)topic ', raw)

    for block in blocks:
        block = block.strip()
        if not block or not block[0].isdigit():
            continue

        lines = [l.strip() for l in block.split("\n") if l.strip()]

        # Extract numeric topic number safely
        match = re.match(r'(\d+)', lines[0])
        if not match:
            continue
        topic_num = int(match.group(1))

        obj = ""
        ts_start = ""
        ts_end = ""

        for line in lines[1:]:
            if line.lower().startswith("objection:"):
                obj = line.split(":", 1)[1].strip()
            elif line.lower().startswith("timestamp_start"):
                ts_start = line.split(":", 1)[1].strip()
            elif line.lower().startswith("timestamp_end"):
                ts_end = line.split(":", 1)[1].strip()

        topics.append({
            "topic_num": topic_num,
            "objection": obj,
            "timestamp_start": ts_start,
            "timestamp_end": ts_end
        })

    return topics


# -----------------------------------------------------
# GPT STAGE 2 — Extract EXACT Muslim Lantern Refutations
# -----------------------------------------------------

def extract_refutations(clean_text: str, stage1_rows: list[dict]) -> str:

    SYSTEM_MESSAGE = """
Extract the EXACT wording of Muslim Lantern's refutation for each Non-Muslim objection.

RULES:
- Use verbatim transcript text (NO paraphrasing).
- Include ALL lines spoken by Muslim Lantern that are part of the refutation.
- Exclude ALL Non-Muslim lines.
- Use objection timestamps to find the refutation section.

Output exactly:

topic 1:
Muslim Lantern refutation:
<exact text>

topic 2:
Muslim Lantern refutation:
<exact text>
"""

    topics_text = "\n".join([f"topic {row['topic_num']}: {row['objection']}" for row in stage1_rows])

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": f"OBJECTIONS:\n{topics_text}\n\nTRANSCRIPT:\n{clean_text}",
            },
        ],
        temperature=0,
    )

    return response.output_text.strip()


# -----------------------------------------------------
# Parse Stage 2 (refutations)
# -----------------------------------------------------

def parse_stage2(raw: str):
    """
    Returns:
    [
        {
            "topic_num": 1,
            "answer": "exact refutation"
        },
        ...
    ]
    """
    results = []
    blocks = re.split(r'(?i)topic ', raw)  # case-insensitive split

    for block in blocks:
        block = block.strip()
        if not block or not block[0].isdigit():
            continue

        lines = [l.strip() for l in block.split("\n") if l.strip()]

        # Extract numeric topic number safely
        match = re.match(r'(\d+)', lines[0])
        if not match:
            continue
        topic_num = int(match.group(1))

        # find "Muslim Lantern refutation:"
        ref_lines = []
        start = False
        for line in lines[1:]:
            if line.lower().startswith("muslim lantern refutation"):
                # may have text after colon
                after = line.split(":", 1)[1].strip()
                if after:
                    ref_lines.append(after)
                start = True
                continue

            if start:
                ref_lines.append(line)

        results.append({
            "topic_num": topic_num,
            "answer": "\n".join(ref_lines).strip()
        })

    return results



# -----------------------------------------------------
# SAVE FINAL MERGED ROWS INTO SUPABASE
# -----------------------------------------------------

def save_topics(video_id: str, stage1_rows: list[dict], stage2_rows: list[dict]):

    # Build lookup for answers
    answers_by_num = {
        row["topic_num"]: row["answer"]
        for row in stage2_rows
    }

    # Merge into final rows
    merged_rows = []
    for row in stage1_rows:
        num = row["topic_num"]
        merged_rows.append({
            "video_id": video_id,
            "topic_num": num,
            "topic": row["objection"],  # Non-Muslim objection
            "answer": answers_by_num.get(num, ""),  # EXACT refutation
            "timestamp_start": row["timestamp_start"],
            "timestamp_end": row["timestamp_end"]
        })

    # Insert
    response = (
        supabase.table("topical_answers")
        .insert(merged_rows)
        .execute()
    )

    if hasattr(response, "error") and response.error:
        raise Exception(f"Error saving topics: {response.error}")

    return response.data
