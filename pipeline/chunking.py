from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def create_chunks_from_topical_answers():
    # 1. Fetch all topical answers
    data = (
        supabase.table("topical_answers")
        .select("id, video_id, topic_num, topic, answer, timestamp_start, timestamp_end")
        .execute()
    )

    rows = data.data
    if not rows:
        print("No topical answers found.")
        return

    chunks_to_insert = []

    # 2. Build chunks for each row
    for row in rows:
        chunk_text = (
            f"Topic: {row['topic']}\n\n"
            f"Answer: {row['answer']}"
        )

        chunk_entry = {
            "video_id": row["video_id"],
            "topical_answer_id": row["id"],
            "topic": row["topic"],
            "topic_num": row["topic_num"],
            "chunk_text": chunk_text
        }

        chunks_to_insert.append(chunk_entry)

    # 3. Insert chunks into Supabase "chunks" table
    if chunks_to_insert:
        insert_result = supabase.table("chunks").insert(chunks_to_insert).execute()
        print("Inserted:", insert_result.data)
    else:
        print("No chunks generated.")


if __name__ == "__main__":
    create_chunks_from_topical_answers()
