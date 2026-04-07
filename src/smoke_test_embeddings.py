#!/usr/bin/env python3
"""
Smoke test for mxbai-embed-large embedding quality on Russian Telegram job posts.

Tests:
  1. Basic: embeddings are correct dimension, non-zero, finite
  2. Russian semantics: similar Russian texts cluster closer than unrelated ones
  3. Telegram lingua: emoji, abbreviations, Cyrillic-Latin mix don't crash/degrade
  4. Live DB: real scam posts are closer to each other than to legit posts

Usage:
    uv run src/smoke_test_embeddings.py
"""

import math
import sys

import psycopg2
import psycopg2.extras
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "mxbai-embed-large"
EMBEDDING_DIM = 1024
DB_DSN = "postgresql://USER:PASS@HOST:PORT/DB"


def embed(text: str) -> list[float]:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": text[:1500]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def main() -> None:
    results = []

    # ── 1. Basic sanity ───────────────────────────────────────────────────────
    print("\n=== 1. Basic sanity ===")

    vec = embed("Работа курьером, оплата ежедневно")
    results.append(check("Correct dimension", len(vec) == EMBEDDING_DIM, f"got {len(vec)}"))
    results.append(check("Non-zero", any(v != 0 for v in vec)))
    results.append(check("All finite", all(math.isfinite(v) for v in vec)))
    results.append(check("Normalized-ish (norm near 1)", 0.9 < math.sqrt(sum(v*v for v in vec)) < 1.1,
                         f"norm={math.sqrt(sum(v*v for v in vec)):.4f}"))

    # ── 2. Russian semantic similarity ───────────────────────────────────────
    print("\n=== 2. Russian semantic similarity ===")

    # Pair A: two similar warehouse job descriptions
    wh1 = "Требуются грузчики на склад, сменный график, оплата 2500р/смена, официальное трудоустройство"
    wh2 = "Вакансия: работник склада, погрузка-разгрузка, з/п от 60 000 руб/мес, оформление по ТК"

    # Pair B: a warehouse job vs an IT job
    it1 = "Ищем Python-разработчика, опыт от 2 лет, FastAPI, PostgreSQL, удаленная работа"

    # Pair C: two easy-money scam posts
    sc1 = "Заработок от 5000р в день! Без опыта, без вложений! Пишите в личку, работа простая"
    sc2 = "Доход 150 000 руб в месяц, гибкий график, работа из дома, пишите в ЛС подробности"

    sim_wh_wh = cosine(embed(wh1), embed(wh2))
    sim_wh_it = cosine(embed(wh1), embed(it1))
    sim_sc_sc = cosine(embed(sc1), embed(sc2))
    sim_wh_sc = cosine(embed(wh1), embed(sc1))

    print(f"  warehouse ↔ warehouse: {sim_wh_wh:.3f}")
    print(f"  warehouse ↔ IT job:    {sim_wh_it:.3f}")
    print(f"  scam ↔ scam:           {sim_sc_sc:.3f}")
    print(f"  warehouse ↔ scam:      {sim_wh_sc:.3f}")

    results.append(check("warehouse clusters with warehouse > IT",
                         sim_wh_wh > sim_wh_it,
                         f"{sim_wh_wh:.3f} > {sim_wh_it:.3f}"))
    results.append(check("scam clusters with scam > warehouse",
                         sim_sc_sc > sim_wh_sc,
                         f"{sim_sc_sc:.3f} > {sim_wh_sc:.3f}"))

    # ── 3. Telegram lingua robustness ────────────────────────────────────────
    print("\n=== 3. Telegram lingua ===")

    telegram_texts = [
        ("emoji-heavy",     "🔥 РАБОТА🔥 Заработок от 3000р/день‼️ Пиши нам👇 без опыта✅"),
        ("abbreviations",   "ЗП 70к/мес, р/ч 400р, ТК РФ, Мск/МО, ПН-ПТ 9-18"),
        ("cyrillic-latin",  "Рaбoтa кyрьeрoм — выcoкий зaрaбoтoк"),  # lookalike substitutions
        ("all-caps",        "ТРЕБУЮТСЯ СРОЧНО РАБОТНИКИ НА СКЛАД ВЫСОКАЯ ОПЛАТА ЗВОНИТЬ"),
        ("very-short",      "Курьер"),
        ("mixed-lang",      "Работа delivery driver, от 80k руб, английский не нужен"),
        ("phone-numbers",   "Звоните: +7 (999) 123-45-67, WhatsApp: 89991234567"),
    ]

    for name, text in telegram_texts:
        try:
            v = embed(text)
            ok = len(v) == EMBEDDING_DIM and all(math.isfinite(x) for x in v)
            results.append(check(f"{name}", ok, f"dim={len(v)}"))
        except Exception as exc:
            results.append(check(f"{name}", False, str(exc)))

    # ── 4. Live DB: real posts ────────────────────────────────────────────────
    print("\n=== 4. Live DB — real posts ===")

    try:
        conn = psycopg2.connect(DB_DSN)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, raw_content, sus_level, sus_category
                FROM jobs
                WHERE embedding IS NOT NULL
                  AND sus_level >= 80
                LIMIT 3
            """)
            high_sus = cur.fetchall()

            cur.execute("""
                SELECT id, raw_content, sus_level, sus_category
                FROM jobs
                WHERE embedding IS NOT NULL
                  AND sus_level <= 15
                LIMIT 3
            """)
            low_sus = cur.fetchall()

        conn.close()

        if not high_sus or not low_sus:
            print("  [SKIP] Not enough embedded rows yet — re-run after backfill")
        else:
            # Embed fresh (not using stored vector, testing round-trip)
            hs_vecs = [embed(r["raw_content"]) for r in high_sus]
            ls_vecs = [embed(r["raw_content"]) for r in low_sus]

            # Avg similarity within high-sus group
            pairs = [(i, j) for i in range(len(hs_vecs)) for j in range(i+1, len(hs_vecs))]
            if pairs:
                avg_within_scam = sum(cosine(hs_vecs[i], hs_vecs[j]) for i, j in pairs) / len(pairs)
            else:
                avg_within_scam = 0.0

            # Avg similarity across groups
            cross_pairs = [(i, j) for i in range(len(hs_vecs)) for j in range(len(ls_vecs))]
            avg_cross = sum(cosine(hs_vecs[i], ls_vecs[j]) for i, j in cross_pairs) / len(cross_pairs)

            print(f"  High-sus (≥80) intra-group similarity: {avg_within_scam:.3f}")
            print(f"  High-sus ↔ low-sus cross similarity:   {avg_cross:.3f}")
            print(f"  Sample high-sus: sus={high_sus[0]['sus_level']} — {high_sus[0]['raw_content'][:80]!r}")
            print(f"  Sample low-sus:  sus={low_sus[0]['sus_level']}  — {low_sus[0]['raw_content'][:80]!r}")

            results.append(check(
                "Scam posts cluster together vs legit",
                avg_within_scam > avg_cross,
                f"within={avg_within_scam:.3f} > cross={avg_cross:.3f}",
            ))

    except Exception as exc:
        print(f"  [SKIP] DB test failed: {exc}")

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Result: {passed}/{total} passed")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
