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
    raise ValueError("SUPABASE_URL is missing — is your .env loaded?")
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
# GPT STAGE 1 — Extract Topics Only
# -----------------------------------------------------

def extract_topics_only(clean_text: str) -> str:

    SYSTEM_MESSAGE = """
You identify ONLY the topics raised in a Muslim Lantern dawah conversation.

RULES:
- Extract ONLY topic titles/questions/claims/objections.
- DO NOT extract Muslim Lantern’s answers.
- DO NOT paraphrase.
- DO NOT add new topics.
- Keep original order.
- Output MUST be:

topic 1: <topic>
topic 2: <topic>
topic 3: <topic>

No explanations. No markdown. No answers.
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
# Parse stage-1 output (extract topic text only)
# -----------------------------------------------------

def parse_topics_list(raw: str):
    topics = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue

        match = re.match(r"topic\s*\d+\s*:\s*(.+)", line, re.IGNORECASE)
        if match:
            topics.append(match.group(1).strip())

    return topics



# -----------------------------------------------------
# GPT STAGE 2 — Extract Muslim Lantern Answers
# -----------------------------------------------------

def extract_answers_for_topics(clean_text: str, topics: list[str]) -> str:

    SYSTEM_MESSAGE = """
Extract EXACT Muslim Lantern responses for each topic.

RULES:
- Use EXACT wording from transcript (no paraphrasing).
- Include ALL lines spoken by Muslim Lantern relevant to the topic.
- Exclude non-Muslim speech completely.
- Keep chronological order.
- Output MUST follow EXACTLY:

topic 1: <topic>
Muslim Lantern answer:
<exact answer>

topic 2: <topic>
Muslim Lantern answer:
<exact answer>
"""

    topics_text = "\n".join([f"- {t}" for t in topics])

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": f"TOPICS:\n{topics_text}\n\nTRANSCRIPT:\n{clean_text}",
            },
        ],
        temperature=0,
    )

    return response.output_text.strip()



# -----------------------------------------------------
# Parse FULL topic + answer output
# -----------------------------------------------------

def parse_topics_output(raw: str):
    lines = [l.strip() for l in raw.split("\n") if l.strip()]

    results = []
    current_topic = None
    current_answer = []

    topic_re = re.compile(r"^topic\s*\d+\s*:\s*(.+)$", re.IGNORECASE)
    answer_re = re.compile(r"^Muslim Lantern answer\s*:\s*(.*)$", re.IGNORECASE)

    i = 0
    while i < len(lines):
        line = lines[i]

        t = topic_re.match(line)
        if t:
            if current_topic and current_answer:
                results.append({
                    "topic": current_topic,
                    "answer": "\n".join(current_answer).strip()
                })
            current_topic = t.group(1).strip()
            current_answer = []
            i += 1
            continue

        a = answer_re.match(line)
        if a:
            first_line = a.group(1).strip()
            if first_line:
                current_answer.append(first_line)

            i += 1
            while i < len(lines):
                if topic_re.match(lines[i]):
                    break
                current_answer.append(lines[i])
                i += 1
            continue

        i += 1

    if current_topic and current_answer:
        results.append({
            "topic": current_topic,
            "answer": "\n".join(current_answer).strip()
        })

    return results



# -----------------------------------------------------
# Save in Supabase
# -----------------------------------------------------

def save_topics(video_id: str, topic_rows: list[dict]):

    # Add video_id to each row
    for row in topic_rows:
        row["video_id"] = video_id

    response = (
        supabase.table("topical_answers")
        .insert(topic_rows)
        .execute()
    )

    if hasattr(response, "error") and response.error:
        raise Exception(f"Error saving topics: {response.error}")

    return response.data


