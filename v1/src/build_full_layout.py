#!/usr/bin/env python3
"""
build_full_layout.py — PaCMAP layout for the full telejobs corpus.

Pulls all embedded jobs from the DB (post-cutoff), runs PaCMAP,
outputs output/full_layout.parquet ready for embedding-atlas.

Usage:
    uv run src/build_full_layout.py
    uv run src/build_full_layout.py --cutoff 2026-02-20 --n-neighbors 15
    uv run src/build_full_layout.py --all   # include pre-cutoff jobs
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

DB_DSN    = "postgresql://USER:PASS@HOST:PORT/DB"
CUTOFF    = "2026-02-20"
OUT_PATH  = Path("output/full_layout.parquet")


def fetch_jobs(conn, cutoff: str | None) -> pd.DataFrame:
    where = "j.embedding IS NOT NULL AND j.sus_factors IS NOT NULL"
    if cutoff:
        where += f" AND j.created_at >= '{cutoff}'"

    sql = f"""
        SELECT
            j.id::text,
            j.sus_level,
            j.sus_category,
            j.work_type,
            j.source_name,
            j.sender_id,
            j.telegram_chat_id,
            j.raw_content,
            j.created_at::text,
            ec.telegram_username AS contact_telegram,
            j.embedding::text
        FROM jobs j
        LEFT JOIN LATERAL (
            SELECT ec2.telegram_username
            FROM job_contacts jc
            JOIN extracted_contacts ec2 ON jc.extracted_contact_id = ec2.id
            WHERE jc.job_id = j.id
            ORDER BY jc.contact_priority ASC NULLS LAST
            LIMIT 1
        ) ec ON true
        WHERE {where}
        ORDER BY j.created_at DESC
    """
    print("Fetching jobs from DB...")
    t0 = time.time()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    print(f"  {len(rows)} rows in {time.time()-t0:.1f}s")
    return rows


def parse_embedding(raw: str) -> list[float]:
    return [float(x) for x in raw.strip("[]").split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default=CUTOFF, help="Min created_at date (default: 2026-02-20)")
    parser.add_argument("--all", action="store_true", help="Include pre-cutoff jobs")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    cutoff = None if args.all else args.cutoff

    conn = psycopg2.connect(DB_DSN)
    rows = fetch_jobs(conn, cutoff)
    conn.close()

    print("Parsing embeddings...")
    t0 = time.time()
    embeddings = np.array([parse_embedding(r["embedding"]) for r in rows], dtype=np.float32)
    print(f"  Matrix: {embeddings.shape} in {time.time()-t0:.1f}s")

    print(f"\nRunning PaCMAP (n_neighbors={args.n_neighbors}, seed={args.seed})...")
    import pacmap
    t0 = time.time()
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        random_state=args.seed,
        verbose=True,
    )
    coords = reducer.fit_transform(embeddings)
    print(f"  Done in {time.time()-t0:.1f}s")

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_norm = (coords[:, 0] - x_min) / (x_max - x_min)
    y_norm = (coords[:, 1] - y_min) / (y_max - y_min)

    print("Building Parquet...")
    records = []
    for i, r in enumerate(rows):
        records.append({
            "id":               r["id"],
            "x":                float(x_norm[i]),
            "y":                float(y_norm[i]),
            "sus_level":        r["sus_level"] or 0,
            "sus_category":     r["sus_category"] or "",
            "work_type":        r["work_type"] or "",
            "source_name":      r["source_name"] or "",
            "sender_id":        r["sender_id"] or "",
            "contact_telegram": r["contact_telegram"] or "",
            "text":             (r["raw_content"] or "")[:300],
            "created_at":       r["created_at"] or "",
        })

    df = pd.DataFrame(records)
    args.out.parent.mkdir(exist_ok=True)
    df.to_parquet(args.out, index=False)
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"\nSaved: {args.out}  ({size_mb:.1f} MB, {len(df)} rows)")
    print(f"\nTo view:  uv run embedding-atlas {args.out} --x x --y y --text text --disable-projection --host 0.0.0.0")


if __name__ == "__main__":
    main()
