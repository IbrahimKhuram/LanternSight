# single_video_tools/extract_topics_one.py

import sys
import traceback
from pipeline.topic_extraction import (
    fetch_clean_transcript,
    extract_objections,
    parse_objections,
    save_topics
)

# ------------------------------------------
# MAIN PIPELINE FUNCTION
# ------------------------------------------

def process_video(video_id: str):

    # ------------------------------------------
    # Step 1 — Fetch Clean Transcript
    # ------------------------------------------

    print(f"\n=== Extracting topics for video: {video_id} ===")
    try:
        clean_text = fetch_clean_transcript(video_id)
        print("✓ Clean transcript fetched")
    except Exception as e:
        print(f"ERROR fetching clean transcript: {e}")
        traceback.print_exc()
        return

    # ------------------------------------------
    # Step 2 — GPT Stage 1 (Extract Objections)
    # ------------------------------------------

    print("Extracting objections from transcript...")
    try:
        stage1_raw = extract_objections(clean_text)
        print("✓ Objection extraction completed")
        print("List of objections (raw GPT output):\n", stage1_raw)
    except Exception as e:
        print(f"ERROR during GPT objection extraction: {e}")
        traceback.print_exc()
        return

    # ------------------------------------------
    # Step 3 — Parse Stage 1 Output Robustly
    # ------------------------------------------

    print("Parsing GPT output...")
    try:
        stage1_rows = parse_objections(stage1_raw)
        print(f"✓ Parsed {len(stage1_rows)} objections")
    except Exception as e:
        print(f"ERROR parsing objections: {e}")
        traceback.print_exc()
        return

    if not stage1_rows:
        print("!!! No objections found — stopping.")
        return

    # ------------------------------------------
    # Step 4 — Save Into Supabase (with refutations)
    # ------------------------------------------

    print("Saving topical answers into Supabase...")
    try:
        result = save_topics(video_id, stage1_rows, clean_text)
        print(f"✓ Saved {len(result)} topical answers into Supabase")
        print("Done.\n")
    except Exception as e:
        print(f"ERROR saving topics: {e}")
        traceback.print_exc()


# ------------------------------------------
# CLI USAGE
# ------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m single_video_tools.extract_topics_one <video_id>")
        sys.exit(1)

    video_id = sys.argv[1].strip()
    process_video(video_id)

