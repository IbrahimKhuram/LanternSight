# single_video_tools/clean_one.py

import sys

from pipeline.cleaning import fetch_raw_transcript, clean_transcript, save_clean_transcript


def clean_one_video(video_id: str):
    print(f"Fetching raw transcript for video {video_id}...")
    raw_text = fetch_raw_transcript(video_id)

    print("Cleaning transcript with GPT...")
    clean_text = clean_transcript(raw_text)

    print("Saving cleaned transcript to Supabase...")
    save_clean_transcript(video_id, clean_text)

    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m single_video_tools.clean_one <video_id>")
        sys.exit(1)

    video_id = sys.argv[1]
    clean_one_video(video_id)
