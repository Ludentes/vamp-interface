#!/usr/bin/env python3
"""
Backfill mxbai-embed-large embeddings for all jobs in the telejobs DB.

Usage:
    uv run src/embed_jobs.py
    uv run src/embed_jobs.py --batch-size 20 --limit 100   # test run

Resumable: skips jobs where embedding IS NOT NULL.
Uses cursor-based pagination (WHERE id > last_id) so errors don't corrupt offset.
"""

import argparse
import sys
import time

import psycopg2
import psycopg2.extras
import requests
from tqdm import tqdm

DB_DSN = "postgresql://USER:PASS@HOST:PORT/DB"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "qwen3-embedding:0.6b"
EMBEDDING_DIM = 1024

# qwen3-embedding supports up to 32k tokens but longer inputs slow inference.
# Telegram posts are typically short; cap at 4000 chars as a safety limit.
MAX_CHARS = 4000


def embed(text: str, retries: int = 3) -> list[float]:
    text = text[:MAX_CHARS]
    for attempt in range(retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


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
    last_id = "00000000-0000-0000-0000-000000000000"

    with tqdm(total=total, unit="job", dynamic_ncols=True) as bar:
        while processed < total:
            batch_size = min(args.batch_size, total - processed)

            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, raw_content
                    FROM jobs
                    WHERE raw_content IS NOT NULL
                      AND raw_content != ''
                      AND embedding IS NULL
                      AND id > %s::uuid
                    ORDER BY id
                    LIMIT %s
                    """,
                    (last_id, batch_size),
                )
                rows = cur.fetchall()

            if not rows:
                break

            updates = []
            for row in rows:
                last_id = str(row["id"])
                try:
                    vec = embed(row["raw_content"])
                    if len(vec) != EMBEDDING_DIM:
                        raise ValueError(f"Unexpected embedding dim: {len(vec)}")
                    updates.append((vec, str(row["id"])))
                except Exception as exc:
                    tqdm.write(f"ERROR job {row['id']}: {exc}")
                    errors += 1

            if updates:
                with conn.cursor() as cur:
                    psycopg2.extras.execute_batch(
                        cur,
                        "UPDATE jobs SET embedding = %s::vector WHERE id = %s::uuid",
                        [(f"[{','.join(map(str, vec))}]", job_id) for vec, job_id in updates],
                    )
                conn.commit()

            n = len(rows)
            processed += n
            bar.update(n)
            bar.set_postfix(ok=len(updates), errors=errors)

    conn.close()
    print(f"\nDone. Processed: {processed}, embedded: {processed - errors}, errors: {errors}")

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
