# single_video_tools/extract_topics_one.py

import sys
from pipeline.topic_extraction import (
    fetch_clean_transcript,
    extract_topics_only,
    parse_topics_list,
    extract_answers_for_topics,
    parse_topics_output,
    save_topics
)

def extract_topics_one(video_id: str):
    print(f"Fetching cleaned transcript for video {video_id}...")
    clean_text = fetch_clean_transcript(video_id)

    # --- STAGE 1: Extract topics only ---
    print("\nRunning GPT Stage 1: Extracting topics...")
    raw_topics = extract_topics_only(clean_text)
    print("\n--- RAW TOPICS OUTPUT ---")
    print(raw_topics)

    # Parse topics into a clean list
    topics_list = parse_topics_list(raw_topics)
    print(f"\nParsed {len(topics_list)} topics:")
    for t in topics_list:
        print(" -", t)

    # --- STAGE 2: Extract Muslim Lantern answers ---
    print("\nRunning GPT Stage 2: Extracting answers...")
    raw_topic_answer_pairs = extract_answers_for_topics(clean_text, topics_list)

    print("\n--- RAW TOPIC + ANSWER PAIRS ---")
    print(raw_topic_answer_pairs)

    # Parse full structured output
    structured = parse_topics_output(raw_topic_answer_pairs)
    print(f"\nStructured topics with answers: {len(structured)} found.")

    # --- SAVE ---
    print("\nSaving into Supabase...")
    save_topics(video_id, structured)

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m single_video_tools.extract_topics_one <video_id>")
        sys.exit(1)

    vid = sys.argv[1]
    extract_topics_one(vid)
