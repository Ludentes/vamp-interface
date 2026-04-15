#!/usr/bin/env python3
"""
score_sus.py — Standalone sus scorer for vamp-interface test dataset.

Replicates the telejobs LLM fraud analysis pipeline:
  1. Send raw_content to Ollama via the same few-shot prompt
  2. Extract 16 sus_factors from JSON response
  3. Compute sus_level via Optuna-calibrated weights
  4. Write updated dataset with new scores + comparison vs originals

Two calibrated weight sets available:
  --weights gemma3   Production weights (optimized for gemma3:12b output)
                     Bucket accuracy 79.1% on 172-example golden set
  --weights gemma4   Experiment weights (optimized for gemma4 output)
                     Bucket accuracy 73.8% — use when running gemma4

Usage:
    uv run src/score_sus.py                          # full dataset, gemma3 weights
    uv run src/score_sus.py --model gemma4:latest    # use gemma4 model + gemma4 weights
    uv run src/score_sus.py --cohorts courier_scam office_legit
    uv run src/score_sus.py --dry-run --limit 5      # preview 5 jobs without writing
    uv run src/score_sus.py --text "Срочно нужен курьер, оплата 5000 в день, пиши в лс"
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# ── Paths ─────────────────────────────────────────────────────────────────────

DATASET_PATH = Path("data/test_dataset.json")
OUTPUT_PATH = Path("data/test_dataset_rescored.json")

OLLAMA_URL = "http://COMFY_HOST:11434"

# ── Prompt (copied verbatim from telejobs fraud_analysis_ru.txt) ───────────────

PROMPT_TEMPLATE = """\
Analyze a Russian Telegram job posting and extract suspicion factors.

Return a JSON object with these exact fields:
- has_specific_address (bool): street + house number present
- has_company_or_org (bool): named organization, IP, or brand
- work_type_clear (bool): concrete task described (not just "courier")
- requires_experience (bool): specific skills/education required
- has_phone_or_address (bool): phone/email for contact (NOT Telegram @username)
- has_salary_info (bool): specific salary/hourly rate with numbers
- only_dm_contact (bool): Telegram DM is sole contact method
- grammar_quality (float 0.0-1.0): 1.0=correct, 0.0=bot-like with Cyrillic substitutions
- urgency_pressure (bool): artificial scarcity ("URGENT! 2 spots left!")
- targets_minors (bool): accepts 14-16 year olds + vague work
- mentions_easy_money (bool): high income without qualification
- no_details_at_all (bool): nearly empty post ("need people, DM" only)
- pay_work_mismatch (bool): suspiciously high pay for unskilled work
- bot_text_patterns (bool): Cyrillic-to-latin letter substitutions
- suspicious_delivery (bool): vague delivery/courier without details
- critical_infrastructure (bool): job involves railroads, cell towers, government buildings, power infrastructure without a named employer (RZD, telecom operator, government agency). Legit jobs from named companies = false. Vague "work" near such objects = true
---
Follow the following format:

Text: ${{text}}
Factors Json: ${{factors_json}}
---

Text: 🌟Персональный ассистент
📊**Зарплата:** от 1000000 до 1500000 KZT
🛠️**Обязанности:**
Административная и организационная деятельность
• Составление и координация ежедневного расписания руководителя;
• Планирование, организация и сопровождение встреч, звонков, деловых поездок;
• Организация мероприятий (личных, корпоративных, имиджевых);
• Контроль исполнения поручений и ведение ежедневной отчётности;
• Сопровождение руководителя на встречах и онлайн-звонках (ведение заметок, фиксация договорённостей, организация follow-up).
Финансовый и операционный менеджмент
• Курирование личных и корпоративных расходов;
• Контроль и оптимизация бюджетов компаний;
• Финансовое планирование, подготовка отчётов, согласование платежей;
• Взаимодействие с бухгалтерами, подрядчиками и партнёрами по финансовым вопросам.
Логистика и партнёрские отношения
• Организация логистики заказов, доставок, командировок и мероприятий;
• Поиск и установление контактов с новыми партнёрами, подрядчиками и поставщиками;
• Ведение переговоров и согласование условий сотрудничества.
Управление проектами и персоналом
• Курирование вопросов, связанных с развитием личного бренда руководителя;
• Координация задач между командами (маркетинг, контент, дизайн, менеджмент проектов);
• Поиск, отбор и координация работы сотрудников и подрядчиков.
🎯**Требования:**
высшее образование
знание английского языка в совершенстве
опыт работы в менеджменте
Factors Json: {{"has_specific_address": false, "has_company_or_org": true, "work_type_clear": true, "requires_experience": true, "has_phone_or_address": false, "has_salary_info": true, "only_dm_contact": true, "grammar_quality": 0.9, "urgency_pressure": false, "targets_minors": false, "mentions_easy_money": false, "no_details_at_all": false, "pay_work_mismatch": false, "bot_text_patterns": false, "suspicious_delivery": false, "critical_infrastructure": false}}

Text: Срочно нужен маляр ( безвоздушная покраска)

Оплата ежедневно 7000

Все вопросы по телефону: 89822358888 Александр
Factors Json: {{"has_specific_address": false, "has_company_or_org": false, "work_type_clear": true, "requires_experience": false, "has_phone_or_address": true, "has_salary_info": true, "only_dm_contact": false, "grammar_quality": 0.9, "urgency_pressure": true, "targets_minors": false, "mentions_easy_money": false, "no_details_at_all": false, "pay_work_mismatch": false, "bot_text_patterns": false, "suspicious_delivery": false, "critical_infrastructure": false}}

Text: 💬 **SMM-специалист в OTUS**

Привет! Мы OTUS — один из крупнейших игроков IT-обучения в России. Обучаем
специалистов по 9 направлениям и 180 программам. За 8 лет помогли 36 тысячам студентов развивать свою карьеру в IT.

Ищем SMM-специалиста, который будет обеспечивать лидогенерацию, растить вовлеченность и подписную базу с помощью соцсетей.

**Что нужно делать**

Полезное действие соцсетей: читай нас 5 минут в день, чтобы системно растить свой грейд как IT-специалиста. Ожидаем, что специалист будет отвечать за рост лидов, подписной базы, охваты и вовлеченность.

Фокусируемся на основных соцсетях VK, TG и YouTube, но хотим развивать и новые площадки.

**Условия**

Работаем удалённо. Оформляем по ТК РФ. Платим 85 000 ₽ на руки + 10 000 ₽ бонус от выполнения плана по лидам после прохождения испытательного срока.

**Как откликнуться**

Отправьте сопроводительное письмо в гугл-форму с рассказом о себе и своём релевантном опыте.
Factors Json: {{"has_specific_address": false, "has_company_or_org": true, "work_type_clear": true, "requires_experience": true, "has_phone_or_address": false, "has_salary_info": true, "only_dm_contact": false, "grammar_quality": 0.9, "urgency_pressure": false, "targets_minors": false, "mentions_easy_money": false, "no_details_at_all": false, "pay_work_mismatch": false, "bot_text_patterns": false, "suspicious_delivery": false, "critical_infrastructure": false}}

Text: Всем привет! 👋 Меня зовут Александр, и я представляю команду "Самокат" 🚴‍♂️

 Приглашаем активных ребят на позиции курьера и сборщика! Идеально, если ищете подработку или хотите совмещать с учебой/работой. (18+)

 🔥 Почему именно "Самокат"? 🔥

 Работа рядом с домом 🏡
 Еженедельные выплаты 💰
 Гибкий график ⏰
 Совмещай с учебой/работой 📚
 Бонус за друзей - 15 000 руб! 🤝
 💰 Сколько можно заработать? 💰

 Хочешь начать зарабатывать уже сейчас? 🚀 Пиши мне в Telegram в личные сообщения! - 89223481782
 Не упусти свой шанс! 😉
Factors Json: {{"has_specific_address": false, "has_company_or_org": true, "work_type_clear": true, "requires_experience": false, "has_phone_or_address": true, "has_salary_info": true, "only_dm_contact": true, "grammar_quality": 0.9, "urgency_pressure": true, "targets_minors": false, "mentions_easy_money": false, "no_details_at_all": false, "pay_work_mismatch": false, "bot_text_patterns": false, "suspicious_delivery": false, "critical_infrastructure": false}}
---

Text: {text}
Factors Json:\
"""

# ── Weight sets ────────────────────────────────────────────────────────────────
# Both sets Optuna-optimized on the telejobs-golden-v2 golden dataset (172 examples).
# Source: telejobs/services/processing-service/experiments/

# Production weights — calibrated on gemma3:12b factor extractions.
# Bucket accuracy 79.1% / MAE 13.0 on golden set.
WEIGHTS_GEMMA3 = {
    "base_score": 43,
    "factors": {
        "has_specific_address": -9,
        "has_company_or_org": -16,
        "work_type_clear": -8,
        "requires_experience": -2,
        "has_phone_or_address": -7,
        "has_salary_info": -7,
        "grammar_quality": -15,
        "no_details_at_all": 23,
        "only_dm_contact": 9,
        "urgency_pressure": 2,
        "targets_minors": 26,
        "mentions_easy_money": 40,
        "pay_work_mismatch": 6,
        "bot_text_patterns": 44,
        "suspicious_delivery": 42,
        "critical_infrastructure": 40,
    },
    "interactions": [
        (["only_dm_contact", "mentions_easy_money"], 25),
        (["no_details_at_all", "mentions_easy_money"], 9),
        (["only_dm_contact", "pay_work_mismatch"], 15),
        (["suspicious_delivery", "only_dm_contact"], 24),
        (["pay_work_mismatch", "urgency_pressure"], 12),
        (["targets_minors", "mentions_easy_money"], 16),
    ],
    "bot_grammar_threshold": 0.2,
    "bot_bonus": 3,
}

# Gemma4 experiment weights — calibrated on gemma4 factor extractions.
# Bucket accuracy 73.8% / MAE 14.5 on golden set.
# Key differences: pay_work_mismatch jumps to 41, grammar_quality drops to 0,
# bot_text_patterns drops to 10. Use with gemma4 model.
WEIGHTS_GEMMA4 = {
    "base_score": 32,
    "factors": {
        "has_specific_address": -1,
        "has_company_or_org": -7,
        "work_type_clear": -11,
        "requires_experience": -19,
        "has_phone_or_address": 0,
        "has_salary_info": -4,
        "grammar_quality": 0,
        "no_details_at_all": 15,
        "only_dm_contact": 13,
        "urgency_pressure": 3,
        "targets_minors": 15,
        "mentions_easy_money": 15,
        "pay_work_mismatch": 41,
        "bot_text_patterns": 10,
        "suspicious_delivery": 33,
        "critical_infrastructure": 40,  # not in experiment, kept from gemma3
    },
    "interactions": [
        (["only_dm_contact", "mentions_easy_money"], 3),
        (["no_details_at_all", "mentions_easy_money"], 22),
        (["only_dm_contact", "pay_work_mismatch"], 18),
        (["suspicious_delivery", "only_dm_contact"], 3),
        (["pay_work_mismatch", "urgency_pressure"], 8),
        (["targets_minors", "mentions_easy_money"], 19),
    ],
    "bot_grammar_threshold": 0.2,
    "bot_bonus": 13,
}


# ── Factor parsing (ported from telejobs llm_fraud_analyzer.py) ───────────────

BOOL_FIELDS = [
    "has_specific_address", "has_company_or_org", "work_type_clear",
    "requires_experience", "has_phone_or_address", "has_salary_info",
    "only_dm_contact", "urgency_pressure", "targets_minors",
    "mentions_easy_money", "no_details_at_all", "pay_work_mismatch",
    "bot_text_patterns", "suspicious_delivery", "critical_infrastructure",
]
FLOAT_FIELDS = ["grammar_quality"]


def parse_factors(data: dict) -> dict[str, Any]:
    """Unwrap nested structures and normalise factor types."""
    for nest_key in ("Factors Json", "factors_json", "Factors", "factors"):
        if nest_key in data and isinstance(data[nest_key], dict):
            data = data[nest_key]
            break
    else:
        if "has_specific_address" not in data:
            nested = [v for v in data.values() if isinstance(v, dict)]
            if len(nested) == 1 and "has_specific_address" in nested[0]:
                data = nested[0]

    factors: dict[str, Any] = {}
    for field in BOOL_FIELDS:
        val = data.get(field)
        if isinstance(val, bool):
            factors[field] = val
        elif isinstance(val, (int, float)):
            factors[field] = bool(val)
    for field in FLOAT_FIELDS:
        val = data.get(field)
        if isinstance(val, (int, float)):
            factors[field] = max(0.0, min(1.0, float(val)))
    return factors


def compute_score(factors: dict[str, Any], weights: dict) -> int | None:
    """Compute sus_level (0-100) from extracted factors + weight set."""
    if not factors:
        return None

    score = weights["base_score"]

    for factor, weight in weights["factors"].items():
        val = factors.get(factor)
        if val is None:
            continue
        if isinstance(val, bool):
            score += weight * int(val)
        elif isinstance(val, float):
            score += weight * val

    for factor_names, bonus in weights["interactions"]:
        if all(factors.get(f) for f in factor_names):
            score += bonus

    grammar = factors.get("grammar_quality")
    if grammar is not None and grammar < weights["bot_grammar_threshold"]:
        score += weights["bot_bonus"]

    return max(0, min(100, int(round(score))))


def sus_category(level: int) -> str:
    if level <= 20:   return "safe"
    if level <= 40:   return "low"
    if level <= 60:   return "medium"
    if level <= 80:   return "high"
    return "critical"


# ── Ollama client ──────────────────────────────────────────────────────────────

async def call_ollama(
    text: str,
    model: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> str:
    """Send text to Ollama, return raw response string."""
    prompt = PROMPT_TEMPLATE.replace("{text}", text)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "top_p": 0.9, "num_predict": 512},
    }
    async with semaphore:
        for attempt in range(retries):
            try:
                r = await client.post("/api/generate", json=payload, timeout=90.0)
                r.raise_for_status()
                return r.json().get("response", "")
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Ollama failed after {retries} attempts: {e}") from e
    return ""


async def score_text(
    text: str,
    model: str,
    weights: dict,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Score a single text. Returns factors + computed sus_level."""
    raw = await call_ollama(text, model, client, semaphore)

    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": f"JSON parse failed: {raw[:200]}", "factors": {}, "sus_level": None}

    factors = parse_factors(data)
    level = compute_score(factors, weights)
    return {
        "factors": factors,
        "sus_level": level,
        "sus_category": sus_category(level) if level is not None else None,
    }


# ── Batch processing ───────────────────────────────────────────────────────────

async def score_dataset(
    jobs: list[dict],
    model: str,
    weights: dict,
    concurrency: int = 4,
    verbose: bool = False,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    done = 0
    t0 = time.monotonic()

    async with httpx.AsyncClient(base_url=OLLAMA_URL) as client:
        async def process(job: dict) -> dict:
            nonlocal done
            # Use full text stored in dataset (up to 2000 chars, same as prod pipeline)
            text = job.get("text", job.get("preview", ""))
            result = await score_text(text, model, weights, client, semaphore)
            done += 1
            elapsed = time.monotonic() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(jobs) - done) / rate if rate > 0 else 0
            orig = job.get("sus_level", "?")
            new = result.get("sus_level", "?")
            delta = ""
            if isinstance(orig, int) and isinstance(new, int):
                d = new - orig
                delta = f" (Δ{d:+d})" if d != 0 else " (=)"
            status = "ERR" if "error" in result else f"{new}{delta}"
            print(f"  [{done:3d}/{len(jobs)}] {job['cohort']:25s} "
                  f"orig={orig:>3}  new={status:<20}  ETA {remaining:.0f}s")
            if verbose and "factors" in result:
                factors = result["factors"]
                fired = []
                for k, v in sorted(factors.items()):
                    w = weights["factors"].get(k, 0)
                    if isinstance(v, bool) and v and w != 0:
                        fired.append(f"{k}({w:+d})")
                    elif isinstance(v, float):
                        fired.append(f"{k}={v:.2f}({w*v:+.1f})")
                print(f"           factors: {', '.join(fired) if fired else '(none triggered)'}")
            return {**job, "rescore": result}

        tasks = [process(job) for job in jobs]
        results = await asyncio.gather(*tasks)

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def print_comparison(rescored: list[dict]) -> None:
    """Print per-cohort before/after statistics."""
    from collections import defaultdict
    cohorts: dict[str, dict] = defaultdict(lambda: {"orig": [], "new": []})

    for job in rescored:
        c = job["cohort"]
        orig = job.get("sus_level")
        new = job.get("rescore", {}).get("sus_level")
        if orig is not None:
            cohorts[c]["orig"].append(orig)
        if new is not None:
            cohorts[c]["new"].append(new)

    print(f"\n{'Cohort':25s}  {'n':>4}  {'orig avg':>8}  {'new avg':>8}  {'drift':>8}  {'new range'}")
    print("─" * 80)
    total_orig, total_new = [], []
    for cohort, data in sorted(cohorts.items()):
        orig_vals = data["orig"]
        new_vals = data["new"]
        if not orig_vals or not new_vals:
            continue
        orig_avg = sum(orig_vals) / len(orig_vals)
        new_avg = sum(new_vals) / len(new_vals)
        drift = new_avg - orig_avg
        new_min, new_max = min(new_vals), max(new_vals)
        total_orig.extend(orig_vals)
        total_new.extend(new_vals)
        print(f"  {cohort:25s}  {len(new_vals):4d}  {orig_avg:8.1f}  {new_avg:8.1f}  "
              f"{drift:+8.1f}  {new_min}-{new_max}")

    print("─" * 80)
    if total_orig and total_new:
        oa = sum(total_orig) / len(total_orig)
        na = sum(total_new) / len(total_new)
        print(f"  {'TOTAL':25s}  {len(total_new):4d}  {oa:8.1f}  {na:8.1f}  {na-oa:+8.1f}")

    # Category shift
    print("\nCategory distribution (original → rescored):")
    cats = ["safe", "low", "medium", "high", "critical"]
    orig_cats: dict[str, int] = {c: 0 for c in cats}
    new_cats: dict[str, int] = {c: 0 for c in cats}
    for job in rescored:
        orig = job.get("sus_level")
        new_cat = job.get("rescore", {}).get("sus_category")
        orig_cat = job.get("sus_category")
        if orig_cat:
            orig_cats[orig_cat] = orig_cats.get(orig_cat, 0) + 1
        if new_cat:
            new_cats[new_cat] = new_cats.get(new_cat, 0) + 1
    for cat in cats:
        o = orig_cats.get(cat, 0)
        n = new_cats.get(cat, 0)
        bar_o = "█" * (o // 5)
        bar_n = "█" * (n // 5)
        print(f"  {cat:10s}: {o:4d} {bar_o:<12}  →  {n:4d} {bar_n}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3:12b",
                        help="Ollama model (default: gemma3:12b)")
    parser.add_argument("--weights", choices=["gemma3", "gemma4", "auto"], default="auto",
                        help="Weight set to use. 'auto' picks based on model name")
    parser.add_argument("--cohorts", nargs="*",
                        help="Only rescore these cohorts (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max jobs to process (for testing)")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Parallel Ollama requests (default: 3)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print which factors fired for each job")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score but don't write output file")
    parser.add_argument("--text", type=str, default=None,
                        help="Score a single text instead of dataset")
    parser.add_argument("--input", type=Path, default=DATASET_PATH,
                        help="Input dataset JSON")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Output dataset JSON")
    args = parser.parse_args()

    # Pick weight set
    if args.weights == "auto":
        weight_set = WEIGHTS_GEMMA4 if "gemma4" in args.model else WEIGHTS_GEMMA3
        weight_name = "gemma4" if "gemma4" in args.model else "gemma3"
    else:
        weight_set = WEIGHTS_GEMMA4 if args.weights == "gemma4" else WEIGHTS_GEMMA3
        weight_name = args.weights

    print(f"Sus scorer")
    print(f"  Model:   {args.model}")
    print(f"  Weights: {weight_name} (base_score={weight_set['base_score']})")
    print(f"  Ollama:  {OLLAMA_URL}")

    # Single-text mode
    if args.text:
        async with httpx.AsyncClient(base_url=OLLAMA_URL) as client:
            sem = asyncio.Semaphore(1)
            result = await score_text(args.text, args.model, weight_set, client, sem)
        print(f"\nText: {args.text[:80]}…")
        print(f"Sus level: {result['sus_level']}  ({result['sus_category']})")
        print("Factors:")
        for k, v in sorted(result["factors"].items()):
            marker = "⚠" if (isinstance(v, bool) and v and
                             weight_set["factors"].get(k, 0) > 0) else " "
            print(f"  {marker} {k:30s}: {v}")
        return

    # Dataset mode
    if not args.input.exists():
        sys.exit(f"Dataset not found: {args.input}\nRun: uv run src/build_test_dataset.py")

    with open(args.input, encoding="utf-8") as f:
        dataset = json.load(f)

    jobs: list[dict] = dataset["jobs"]

    if args.cohorts:
        jobs = [j for j in jobs if j["cohort"] in args.cohorts]
        print(f"  Cohorts: {args.cohorts} ({len(jobs)} jobs)")

    if args.limit:
        jobs = jobs[:args.limit]

    print(f"  Jobs:    {len(jobs)}\n")

    if args.dry_run:
        print("[DRY RUN] Would score the above jobs. Showing 3 sample calls:\n")
        jobs = jobs[:3]

    rescored = await score_dataset(
        jobs, args.model, weight_set,
        concurrency=args.concurrency, verbose=args.verbose,
    )

    print_comparison(rescored)

    if args.dry_run:
        print("\n[DRY RUN] Output not written.")
        return

    # Merge rescored back into full dataset
    rescored_by_id = {j["id"]: j for j in rescored}
    merged_jobs = []
    for job in dataset["jobs"]:
        if job["id"] in rescored_by_id:
            merged_jobs.append(rescored_by_id[job["id"]])
        else:
            merged_jobs.append(job)

    out = {
        **dataset,
        "rescored_with": {"model": args.model, "weights": weight_name},
        "jobs": merged_jobs,
    }

    args.output.parent.mkdir(exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"\nSaved: {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    asyncio.run(main())
