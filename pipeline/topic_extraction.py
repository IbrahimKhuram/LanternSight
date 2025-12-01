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

    # supabase client returns dict-like object with .data or .error
    if getattr(response, "data", None) is None:
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
- Identify timestamp_end = when Muslim Lantern moves on to a different topic AFTER COMPLETING refuting that objection.

Output format (examples - be consistent):

objection 1: <non-muslim objection>
timestamp_start: 00:12
timestamp_end: 01:40

objection 2: <non-muslim objection>
timestamp_start: 02:05
timestamp_end: 03:10
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

    # response.output_text should contain the model text
    return response.output_text.strip()


# -----------------------------------------------------
# Parse Stage 1 Output (Robust)
# -----------------------------------------------------

def parse_objections(raw: str):
    """
    Robust parser for multiple GPT output variations.

    Returns:
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

    if not raw:
        return []

    # Normalize common variants so the parser can find sections reliably
    normalized = raw.replace("Objection #", "objection ").replace("Topic #", "topic ")
    normalized = normalized.replace("Topic", "objection")  # be forgiving
    # Collapse windows CRLF to \n
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Find all headings like: "objection 1", "objection 1:", "objection 1 -", "objection 1)"
    heading_re = re.compile(r'(?i)\b(?:objection|topic)\s*#?\s*(\d+)\b[.: -]*')

    matches = list(heading_re.finditer(normalized))
    if not matches:
        # Maybe the model output uses plain "objection:" repeated without numbers.
        # Try splitting by lines starting with "objection" and assign sequential numbers.
        lines = [l.strip() for l in normalized.split("\n") if l.strip()]
        blocks = []
        cur = []
        for line in lines:
            if re.match(r'(?i)^objection\b', line):
                if cur:
                    blocks.append("\n".join(cur))
                cur = [line]
            else:
                cur.append(line)
        if cur:
            blocks.append("\n".join(cur))

        results = []
        num = 1
        for block in blocks:
            parsed = _parse_single_objection_block(block, num)
            results.append(parsed)
            num += 1
        return results

    # Build blocks by heading spans (from end of this heading to start of next heading)
    blocks = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized)
        blocks.append((m.group(1), normalized[start:end].strip()))

    results = []
    used_topic_nums = set()
    max_assigned = 0

    for num_str, block_text in blocks:
        try:
            topic_num = int(num_str)
        except Exception:
            # fallback to sequential numbering
            max_assigned += 1
            topic_num = max_assigned

        # ensure uniqueness: if duplicate number, bump to next available integer
        if topic_num in used_topic_nums:
            # find next available
            candidate = max(used_topic_nums) + 1 if used_topic_nums else 1
            while candidate in used_topic_nums:
                candidate += 1
            topic_num = candidate

        used_topic_nums.add(topic_num)
        max_assigned = max(max_assigned, topic_num)

        parsed = _parse_single_objection_block(block_text, topic_num)
        results.append(parsed)

    return results


def _parse_single_objection_block(block: str, topic_num: int) -> dict:
    """
    Parse a single block of text (the content after the 'objection N' heading)
    and extract objection text, timestamp_start, timestamp_end.
    """
    objection = ""
    ts_start = ""
    ts_end = ""

    # Split into lines and look for common keys
    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue

        low = line.lower()
        # Objection line could be "objection 1: Why ...", or "objection: Why..."
        if low.startswith("objection"):
            # split on first colon (if any), otherwise take the remainder of line
            if ":" in line:
                objection = line.split(":", 1)[1].strip()
            else:
                # line might be only 'objection' and next lines are text
                objection = line[len("objection"):].strip()

        # Common timestamp labels
        elif ("timestamp_start" in low or "start:" in low) and re.search(r'\d{1,2}:\d{2}', line):
            ts_match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)', line)
            if ts_match:
                ts_start = ts_match.group(1)

        elif ("timestamp_end" in low or "end:" in low) and re.search(r'\d{1,2}:\d{2}', line):
            ts_match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)', line)
            if ts_match:
                ts_end = ts_match.group(1)


        else:
            # fallback heuristics:
            # If line contains a timestamp and we don't yet have start, use it for start.
            ts_match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)', line)
            if ts_match and not ts_start:
                ts_start = ts_match.group(1)
            # If line contains "to" or "-" between timestamps like "00:12 - 01:45" or "00:12 to 01:45"
            range_match = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:-|to|–)\s*(\d{1,2}:\d{2}(?::\d{2})?)', line, flags=re.I)
            if range_match:
                ts_start = range_match.group(1)
                ts_end = range_match.group(2)

    # If objection is still empty, maybe the first non-empty line is the objection text:
    if not objection:
        # take first non-empty line in block that is not a timestamp-only line
        for line in block.split("\n"):
            if not line.strip():
                continue
            if re.search(r'\d{1,2}:\d{2}', line) and len(line.strip()) < 12:
                # skip lines that are only timestamps
                continue
            objection = line.strip()
            break

    # final cleanup: strip any leading/trailing punctuation
    objection = objection.strip(" \t\n\r-:") if objection else ""
    ts_start = ts_start.strip() if ts_start else ""
    ts_end = ts_end.strip() if ts_end else ""

    return {
        "topic_num": topic_num,
        "objection": objection,
        "timestamp_start": ts_start,
        "timestamp_end": ts_end
    }


# -----------------------------------------------------
# NEW: Extract EXACT Muslim Lantern Refutation from transcript
# -----------------------------------------------------

# allow optional brackets around timestamps and optional leading spaces
# match 0:00 or 00:00 or 0:00:00 etc
TS_SEARCH_PATTERN = r'(\d{1,2}:\d{2}(?::\d{2})?)'

def timestamp_to_seconds(ts: str) -> int:
    ts = ts.strip()
    # If ts contains a range like "00:12 - 01:20", take the first part
    if "-" in ts:
        ts = ts.split("-")[0].strip()
    if "to" in ts.lower():
        ts = ts.lower().split("to")[0].strip()

    parts = [int(x) for x in ts.split(":")]
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    raise ValueError(f"Invalid timestamp: {ts}")


def extract_refutation_from_transcript(clean_text: str, ts_start: str, ts_end: str) -> str:
    """
    Extracts ONLY Muslim Lantern lines within the timestamp window.
    Removes:
      - timestamps (only at start of line)
      - speaker labels
      - Non-Muslim lines
    """

    if not ts_start or not ts_end:
        return ""

    start_sec = timestamp_to_seconds(ts_start)
    end_sec = timestamp_to_seconds(ts_end)

    extracted = []

    for line in clean_text.split("\n"):
        raw_line = line.strip()
        if not raw_line:
            continue

        # find first timestamp in the line anywhere
        ts_match = re.search(TS_SEARCH_PATTERN, raw_line)
        if not ts_match:
            continue

        ts = ts_match.group(1)
        try:
            ts_sec = timestamp_to_seconds(ts)
        except ValueError:
            # skip lines with malformed timestamps
            continue

        # must be inside time range
        if not (start_sec <= ts_sec <= end_sec):
            continue

        # must be Muslim Lantern speaking (case-insensitive, allow colon or dash)
        # e.g. "Muslim Lantern:", "Muslim Lantern -", "Muslim Lantern:"
        if re.search(r'(?i)\bMuslim Lantern\b\s*[:\-–]', raw_line) is None:
            # also handle variations like "Muslim Lantern:" with no punctuation by checking whole line after timestamp
            # but prefer exact match
            continue

        # remove the timestamp only at start (or at first occurrence) to avoid chopping in-line references
        # remove the matched timestamp occurrence (the first one)
        cleaned = re.sub(r'^\s*\[?' + TS_SEARCH_PATTERN + r'\]?\s*', "", raw_line).strip()


        # remove the speaker label (case-insensitive, only the first occurrence)
        cleaned = re.sub(r'(?i)\bMuslim Lantern\b\s*[:\-–]?\s*', "", cleaned, count=1).strip()

        if cleaned:
            extracted.append(cleaned)

    # join by newline and return
    return "\n".join(extracted).strip()


# -----------------------------------------------------
# SAVE FINAL ROWS INTO SUPABASE
# -----------------------------------------------------

def save_topics(video_id: str, stage1_rows: list[dict], clean_text: str):
    final_rows = []

    for row in stage1_rows:
        answer_text = extract_refutation_from_transcript(
            clean_text,
            row.get("timestamp_start", ""),
            row.get("timestamp_end", "")
        )

        final_rows.append({
            "video_id": video_id,
            "topic_num": row["topic_num"],
            "topic": row["objection"],
            "answer": answer_text,
            "timestamp_start": row["timestamp_start"],
            "timestamp_end": row["timestamp_end"]
        })

    response = (
        supabase.table("topical_answers")
        .insert(final_rows)
        .execute()
    )

    # supabase client typically sets .error if something went wrong
    if getattr(response, "error", None):
        raise Exception(f"Error saving topics: {response.error}")

    return response.data
