#!/usr/bin/env python3
"""
build_test_dataset.py — Build a curated test dataset from telejobs DB.

Selection criteria:
  - created_at >= 2026-02-20 (sus detection model was unreliable before this)
  - embedding IS NOT NULL
  - sus_factors IS NOT NULL
  - Balanced cohorts across work types and fraud levels

Output: data/test_dataset.json with all metadata + embeddings inline.

Usage:
    uv run src/build_test_dataset.py
    uv run src/build_test_dataset.py --dry-run    # print cohort stats only
    uv run src/build_test_dataset.py --n 30       # 30 per cohort (default 50)
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import psycopg2
import psycopg2.extras

DB_DSN = "postgresql://USER:PASS@HOST:PORT/DB"
DATA_DIR = Path("data")
CUTOFF_DATE = "2026-02-20"
RANDOM_SEED = 42

# Cohort definitions. Each entry:
#   name, work_type (None = any), sus_min, sus_max, extra_where (SQL fragment), label
COHORTS = [
    # ── Legitimate clusters ──────────────────────────────────────────────────
    {
        "name": "warehouse_legit",
        "work_type": "склад",
        "sus_min": 0, "sus_max": 30,
        "extra_where": None,
        "target_n": 50,
        "fraud": False,
    },
    {
        "name": "construction_legit",
        "work_type": "стройка",
        "sus_min": 0, "sus_max": 30,
        "extra_where": None,
        "target_n": 50,
        "fraud": False,
    },
    {
        "name": "cleaning_legit",
        "work_type": "уборка",
        "sus_min": 0, "sus_max": 30,
        "extra_where": None,
        "target_n": 50,
        "fraud": False,
    },
    {
        "name": "office_legit",
        "work_type": "офис",
        "sus_min": 0, "sus_max": 30,
        "extra_where": None,
        "target_n": 50,
        "fraud": False,
    },
    # ── Fraud clusters ───────────────────────────────────────────────────────
    {
        "name": "courier_scam",
        "work_type": "доставка",
        "sus_min": 70, "sus_max": 100,
        "extra_where": None,
        "target_n": 50,
        "fraud": True,
    },
    {
        "name": "remote_scam",
        "work_type": "удалёнка",
        "sus_min": 60, "sus_max": 100,
        "extra_where": None,
        "target_n": 50,
        "fraud": True,
    },
    {
        "name": "office_scam",
        "work_type": "офис",
        "sus_min": 60, "sus_max": 100,
        "extra_where": None,
        "target_n": 40,  # smaller pool
        "fraud": True,
    },
    {
        "name": "warehouse_scam",
        "work_type": "склад",
        "sus_min": 60, "sus_max": 100,
        "extra_where": None,
        "target_n": 40,  # smaller pool
        "fraud": True,
    },
    # ── Factor-based fraud (cross-work-type) ─────────────────────────────────
    {
        "name": "easy_money_scam",
        "work_type": None,
        "sus_min": 80, "sus_max": 100,
        "extra_where": "(sus_factors->>'mentions_easy_money')::boolean = true",
        "target_n": 50,
        "fraud": True,
    },
    {
        "name": "pay_mismatch_scam",
        "work_type": None,
        "sus_min": 80, "sus_max": 100,
        "extra_where": "(sus_factors->>'pay_work_mismatch')::boolean = true",
        "target_n": 50,
        "fraud": True,
    },
    # ── Medium sus band (thin, but important for gradient testing) ────────────
    {
        "name": "medium_sus",
        "work_type": None,
        "sus_min": 35, "sus_max": 65,
        "extra_where": None,
        "target_n": 50,
        "fraud": None,  # ambiguous ground truth
    },
    # ── High-medium band (corpus thin zone: 65–89) ────────────────────────────
    {
        "name": "high_medium_sus",
        "work_type": None,
        "sus_min": 65, "sus_max": 89,
        "extra_where": None,
        "target_n": 30,  # pool is small (~1200 total)
        "fraud": None,
    },
]


def build_query(cohort: dict, limit_factor: int = 5) -> tuple[str, list]:
    """Build SELECT query for a cohort. Fetches limit_factor × target_n rows for random sampling."""
    conditions = [
        "j.embedding IS NOT NULL",
        "j.sus_factors IS NOT NULL",
        f"j.created_at >= '{CUTOFF_DATE}'",
        "j.sus_level >= %s",
        "j.sus_level <= %s",
    ]
    params: list = [cohort["sus_min"], cohort["sus_max"]]

    if cohort.get("work_type"):
        conditions.append("j.work_type = %s")
        params.append(cohort["work_type"])

    if cohort.get("extra_where"):
        conditions.append(cohort["extra_where"].replace("sus_factors", "j.sus_factors"))

    limit = cohort["target_n"] * limit_factor
    params.append(limit)

    sql = f"""
        SELECT j.id::text, j.raw_content, j.sus_level, j.sus_category, j.work_type,
               j.sus_factors, j.embedding, j.created_at::text,
               j.source_name, j.sender_id, j.telegram_chat_id,
               ec.telegram_username AS contact_telegram,
               ec.phone_hash        AS contact_phone_hash
        FROM jobs j
        LEFT JOIN LATERAL (
            SELECT ec2.telegram_username, ec2.phone_hash
            FROM job_contacts jc
            JOIN extracted_contacts ec2 ON jc.extracted_contact_id = ec2.id
            WHERE jc.job_id = j.id
            ORDER BY jc.contact_priority ASC NULLS LAST
            LIMIT 1
        ) ec ON true
        WHERE {' AND '.join(conditions)}
        ORDER BY j.created_at DESC
        LIMIT %s
    """
    return sql, params


def parse_embedding(raw) -> list[float]:
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",")]
    return list(raw)


def sample_cohort(conn, cohort: dict, rng: random.Random) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        sql, params = build_query(cohort)
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        return []

    # Random sample without replacement
    population = list(rows)
    rng.shuffle(population)
    selected = population[:cohort["target_n"]]

    result = []
    for row in selected:
        result.append({
            "id": row["id"],
            "cohort": cohort["name"],
            "fraud": cohort["fraud"],
            "sus_level": row["sus_level"],
            "sus_category": row["sus_category"],
            "work_type": row["work_type"],
            "sus_factors": dict(row["sus_factors"]) if row["sus_factors"] else {},
            "embedding": parse_embedding(row["embedding"]),
            "text": (row["raw_content"] or "")[:2000],
            "created_at": row["created_at"],
            "source_name": row["source_name"],
            "sender_id": row["sender_id"],
            "telegram_chat_id": row["telegram_chat_id"],
            "contact_telegram": row["contact_telegram"],
            "contact_phone_hash": row["contact_phone_hash"],
        })

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print stats, don't write file")
    parser.add_argument("--n", type=int, default=None, help="Override target_n for all cohorts")
    args = parser.parse_args()

    if args.n:
        for c in COHORTS:
            c["target_n"] = args.n

    rng = random.Random(RANDOM_SEED)
    conn = psycopg2.connect(DB_DSN)

    print(f"Building test dataset (cutoff: {CUTOFF_DATE}, seed: {RANDOM_SEED})\n")

    all_jobs: list[dict] = []
    seen_ids: set[str] = set()

    for cohort in COHORTS:
        rows = sample_cohort(conn, cohort, rng)

        # Deduplicate across cohorts (easy_money/pay_mismatch may overlap)
        unique = [r for r in rows if r["id"] not in seen_ids]
        for r in unique:
            seen_ids.add(r["id"])

        all_jobs.extend(unique)
        sus_vals = [r["sus_level"] for r in unique]
        print(f"  {cohort['name']:25s}: {len(unique):3d} jobs | "
              f"sus {min(sus_vals) if sus_vals else '?'}-{max(sus_vals) if sus_vals else '?'} | "
              f"fraud={'yes' if cohort['fraud'] else 'no' if cohort['fraud'] is False else 'mixed'}")

    conn.close()

    print(f"\nTotal: {len(all_jobs)} jobs ({len(seen_ids)} unique)")
    print(f"\nBreakdown:")
    legit = [j for j in all_jobs if j["fraud"] is False]
    fraud = [j for j in all_jobs if j["fraud"] is True]
    mixed = [j for j in all_jobs if j["fraud"] is None]
    print(f"  Legit:  {len(legit)}")
    print(f"  Fraud:  {len(fraud)}")
    print(f"  Medium: {len(mixed)}")
    print(f"\nSus distribution:")
    buckets = {}
    for j in all_jobs:
        b = (j["sus_level"] // 10) * 10
        buckets[b] = buckets.get(b, 0) + 1
    for b in sorted(buckets):
        bar = "#" * (buckets[b] // 2)
        print(f"  sus {b:3d}-{b+9}: {buckets[b]:4d}  {bar}")

    if args.dry_run:
        print("\n[DRY RUN] No file written.")
        return

    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "test_dataset.json"

    manifest = {
        "built_at": datetime.now().isoformat(),
        "cutoff_date": CUTOFF_DATE,
        "random_seed": RANDOM_SEED,
        "total": len(all_jobs),
        "cohorts": {
            c["name"]: len([j for j in all_jobs if j["cohort"] == c["name"]])
            for c in COHORTS
        },
        "jobs": all_jobs,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
