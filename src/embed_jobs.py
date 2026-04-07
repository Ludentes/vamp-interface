#!/usr/bin/env python3
"""
Backfill mxbai-embed-large embeddings for all jobs in the telejobs DB.

Usage:
    uv run src/embed_jobs.py
    uv run src/embed_jobs.py --batch-size 20 --limit 100   # test run

Resumable: skips jobs where embedding IS NOT NULL.
"""

import argparse
import sys
import psycopg2
import psycopg2.extras
import requests
from tqdm import tqdm

DB_DSN = "postgresql://USER:PASS@HOST:PORT/DB"
OLLAMA_URL = "http://COMFY_HOST:11434/api/embeddings"
OLLAMA_MODEL = "mxbai-embed-large"
EMBEDDING_DIM = 1024


def embed(text: str) -> list[float]:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Stop after N jobs (for testing)")
    args = parser.parse_args()

    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM jobs WHERE raw_content IS NOT NULL AND raw_content != '' AND embedding IS NULL"
        )
        row = cur.fetchone()
        total = row[0] if row else 0

    if total == 0:
        print("All jobs already embedded.")
        conn.close()
        return

    if args.limit:
        total = min(total, args.limit)

    print(f"Jobs to embed: {total}")

    processed = 0
    errors = 0
    offset = 0

    with tqdm(total=total, unit="job", dynamic_ncols=True) as bar:
        while processed < total:
            batch_size = min(args.batch_size, total - processed)

            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, raw_content
                    FROM jobs
                    WHERE raw_content IS NOT NULL AND raw_content != '' AND embedding IS NULL
                    ORDER BY created_at
                    LIMIT %s OFFSET %s
                    """,
                    (batch_size, offset),
                )
                rows = cur.fetchall()

            if not rows:
                break

            updates = []
            for row in rows:
                try:
                    vec = embed(row["raw_content"])
                    if len(vec) != EMBEDDING_DIM:
                        raise ValueError(f"Unexpected embedding dim: {len(vec)}")
                    updates.append((vec, str(row["id"])))
                except Exception as e:
                    tqdm.write(f"ERROR job {row['id']}: {e}")
                    errors += 1
                    offset += 1  # skip this job next iteration
                    continue

            if updates:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_batch(
                        cur,
                        "UPDATE jobs SET embedding = %s::vector WHERE id = %s::uuid",
                        [(f"[{','.join(map(str, vec))}]", job_id) for vec, job_id in updates],
                    )
                conn.commit()

            n = len(updates)
            processed += n
            bar.update(n)
            bar.set_postfix(errors=errors)

    conn.close()
    print(f"\nDone. Embedded: {processed}, errors: {errors}")

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
