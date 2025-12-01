# pipeline/cleaning.py
import os
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

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------
# Functions
# ----------------------------

def fetch_raw_transcript(video_id: str) -> str:
    """Fetch raw transcript from Supabase for a video_id"""
    response = (
        supabase.table("raw_transcripts")
        .select("raw_text")
        .eq("video_id", video_id)
        .single()
        .execute()
    )

    if not response.data:
        raise Exception(f"No raw transcript found for video_id: {video_id}")

    return response.data["raw_text"]

def clean_transcript(raw_text: str) -> str:
    """
    Use GPT-4.1 to clean the transcript, preserve timestamps, fix errors,
    and label speakers using the new OpenAI Responses API.
    """

    SYSTEM_INSTRUCTIONS = """
You are processing a raw, messy dialogue transcript from a Muslim Lantern dawah video. 
Your task is to output a clean, readable, and timestamped conversation with accurate speaker turns.

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
- Keep all timestamps exactly as they appear. If a timestamp is present, attach it to that line.

Output format (MUST FOLLOW EXACTLY):

[timestamp] Speaker: cleaned dialogue line
[timestamp] Speaker: cleaned dialogue line
[timestamp] Speaker: cleaned dialogue line
(continue)

"""

    response = client.responses.create(
        model="gpt-4.1", #gpt-5
        #reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": f"RAW TRANSCRIPT:\n{raw_text}"}
        ],
        temperature=0
    )

    clean_text = response.output_text.strip()
    return clean_text



def save_clean_transcript(video_id: str, clean_text: str):
    """Save cleaned transcript to Supabase"""
    data = {
        "video_id": video_id,
        "clean_text": clean_text
    }

    response = supabase.table("formatted_transcripts").insert(data).execute()

    if hasattr(response, "error") and response.error:
        raise Exception(f"Error saving formatted transcript: {response.error}")

    return response.data

