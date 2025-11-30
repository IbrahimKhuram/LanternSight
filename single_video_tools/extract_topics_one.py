# single_video_tools/extract_topics_one.py

import sys
from pipeline.topic_extraction import (
    fetch_clean_transcript,
    extract_objections,
    parse_stage1,
    extract_refutations,
    parse_stage2,
    save_topics,
)

def process_one(video_id: str):
    print(f"\n--- Extracting topics for video_id = {video_id} ---")

    # ------------------------------------------
    # 1. Fetch cleaned transcript
    # ------------------------------------------
    print("Fetching cleaned transcript...")
    clean_text = fetch_clean_transcript(video_id)

    # ------------------------------------------
    # 2. Stage 1 — Extract objections + timestamps
    # ------------------------------------------
    print("Running GPT Stage 1 (objection extraction)...")
    stage1_raw = extract_objections(clean_text)
    stage1_rows = parse_stage1(stage1_raw)

    print(f"Found {len(stage1_rows)} objections.")

    # Print objections with timestamps
    print("Objections with timestamps:")
    for row in stage1_rows:
        print(f"- Topic {row['topic_num']}: {row['objection']}")
        print(f"  Start: {row['timestamp_start']}, End: {row['timestamp_end']}\n")

    # ------------------------------------------
    # 3. Stage 2 — Extract exact ML refutations
    # ------------------------------------------
    print("Running GPT Stage 2 (refutation extraction)...")
    stage2_raw = extract_refutations(clean_text, stage1_rows)

    # Print raw Stage 2 output
    print("\n--- Raw Stage 2 Output ---\n")
    print(stage2_raw)
    print("\n--- End of Raw Stage 2 Output ---\n")

    stage2_rows = parse_stage2(stage2_raw)

    # Print structured Stage 2 output
    print("Structured Stage 2 Output:")
    for row in stage2_rows:
        print(f"- Topic {row['topic_num']}:")
        print(f"{row['answer']}\n")

    # ------------------------------------------
    # 4. Save to Supabase
    # ------------------------------------------
    print("Saving topics to Supabase...")
    save_topics(video_id, stage1_rows, stage2_rows)

    print("\n✓ DONE — Topics successfully extracted and saved.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m single_video_tools.extract_topics_one <video_id>")
        sys.exit(1)

    video_id = sys.argv[1]
    process_one(video_id)
