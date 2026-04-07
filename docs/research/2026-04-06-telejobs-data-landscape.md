# Telejobs Data Landscape for Semantic Visualization

**Date:** 2026-04-06  
**Purpose:** Inventory of available data fields, computed factors, and what can be added to support semantic/visual exploration of job postings.

---

## What We Already Have

### 1. Structured Metadata (DB, ready to use)

These are extracted per job and stored in the `jobs` table.

| Field | Type | Notes |
|-------|------|-------|
| `title` | str | Job title |
| `company` | str | Organization name |
| `location` | str | City/region |
| `is_remote` | bool | |
| `job_type` | str | full_time / part_time / contract / gig |
| `work_type` | str | Russian category: склад, доставка, стройка, etc. |
| `experience_level` | str | junior / mid / senior |
| `salary_min` / `salary_max` | Decimal | |
| `salary_currency` | str | RUB, USD, EUR |
| `salary_period` | str | monthly / daily / hourly |
| `payout` | str | daily / weekly / monthly / on_completion |
| `positions_available` | int | |
| `required_skills` | str[] | Array of skills |
| `contact_method` | str | |
| `source_name` | str | Telegram channel name |
| `posted_at` | datetime | Original message timestamp |
| `extraction_confidence` | float | 0–1, quality of LLM extraction |

### 2. SUS Detection Results (DB, ready to use)

Stored in `jobs` (summary) and `sus_detection_log` (full record).

| Field | Type | Notes |
|-------|------|-------|
| `sus_level` | int 0–100 | Final fraud score |
| `sus_category` | str | safe / low / medium / high / critical |
| `sus_confidence` | float | Confidence in the score |
| `sus_detection_method` | str | rules / llm / hybrid / failed |
| `sus_reasons` | str[] | Triggered rule names |
| `rules_triggered` | JSONB | Full rule match details |

### 3. Plan B — LLM-Extracted Factors (DB, `sus_factors` JSONB)

16 factors extracted from job text per run. Stored raw in JSONB. The most semantically rich per-job signal we have. Any factor can be `null` if the LLM omitted it — treated as "not present" (skipped in score computation, not the same as `false`).

**Legitimacy signals** — `true` lowers suspicion score:

| Factor | Type | Range | Weight | Description |
|--------|------|-------|--------|-------------|
| `has_specific_address` | bool | `{false, true}` | −9 | Concrete street/location given |
| `has_company_or_org` | bool | `{false, true}` | −16 | Named organization mentioned |
| `work_type_clear` | bool | `{false, true}` | −9 | Specific tasks described (not just "courier") |
| `requires_experience` | bool | `{false, true}` | −5 | Skills or qualifications required |
| `has_phone_or_address` | bool | `{false, true}` | −10 | Non-Telegram contact info present |
| `has_salary_info` | bool | `{false, true}` | −7 | Concrete salary amount stated |
| `grammar_quality` | float | `[0.0, 1.0]` | −15 | 1.0 = fluent; <0.2 triggers +3 bot-text bonus |

**Fraud signals** — `true` raises suspicion score:

| Factor | Type | Range | Weight | Description |
|--------|------|-------|--------|-------------|
| `no_details_at_all` | bool | `{false, true}` | +23 | Almost no job details in posting |
| `only_dm_contact` | bool | `{false, true}` | +9 | No contact method outside Telegram DM |
| `urgency_pressure` | bool | `{false, true}` | +2 | Artificial urgency ("today only", "limited slots") |
| `targets_minors` | bool | `{false, true}` | +26 | Aimed at under-18 workers |
| `mentions_easy_money` | bool | `{false, true}` | +40 | "From 5000/day, no experience needed" |
| `pay_work_mismatch` | bool | `{false, true}` | +6 | Pay is suspiciously high for the described work |
| `bot_text_patterns` | bool | `{false, true}` | +40 | Cyrillic-Latin character substitutions detected |
| `suspicious_delivery` | bool | `{false, true}` | +42 | Vague courier/delivery framing, no named company |
| `critical_infrastructure` | bool | `{false, true}` | +40 | Infrastructure-adjacent work, no employer identified |

**Score formula:** `sus_score = 43 (base) + Σ(weight × factor) + interaction_bonuses`, clamped to `[0, 100]`.

**Interaction bonuses** (applied when both factors are `true`):

| Factor pair | Bonus |
|-------------|-------|
| `only_dm_contact` + `mentions_easy_money` | +25 |
| `suspicious_delivery` + `only_dm_contact` | +24 |
| `targets_minors` + `mentions_easy_money` | +16 |
| `only_dm_contact` + `pay_work_mismatch` | +15 |
| `pay_work_mismatch` + `urgency_pressure` | +10 |
| `no_details_at_all` + `mentions_easy_money` | +9 |

These 15 booleans + 1 float form a **16-dimensional factor vector** per job — directly usable for clustering and visualization.

### 4. Rule Engine — Pattern-Matching Signals (DB, `rules_triggered` JSONB)

10 keyword/regex rules from `config/sus_rules.yaml` v1.4. Each fired rule contributes to `sus_level` via: `rule_score = base_score × rule_weight × category_weight`, summed and capped at 100.

| Rule ID | Category | Base Score | Rule Weight | Category Weight | Effective Range | Severity |
|---------|----------|-----------|-------------|-----------------|-----------------|----------|
| `anonymous_suspicious_delivery` | trafficking | 70 | 2.0 | 2.0 | 0–280* | critical |
| `suspicious_contact_methods` | contact | 50 | 2.0 | 1.2 | 0–120* | critical |
| `data_collection_scam` | contact | 45 | 1.8 | 1.2 | 0–97* | critical |
| `high_pay_anonymous_courier` | trafficking | 60 | 1.8 | 2.0 | 0–216* | critical |
| `one_time_quick_money` | payment | 40 | 1.5 | 1.5 | 0–90* | critical |
| `unrealistic_benefits` | content | 35 | 1.4 | 1.0 | 0–49 | high |
| `high_salary_no_qualification` | payment | 30 | 1.2 | 1.5 | 0–54* | high |
| `mlm_pyramid_indicators` | content | 30 | 1.3 | 1.0 | 0–39 | high |
| `urgency_pressure` | urgency | 20 | 1.3 | 1.3 | 0–34 | medium |
| `details_only_after_contact` | content | 25 | 1.1 | 1.0 | 0–28 | medium |

\* Capped at 100 after all rules are summed.

**Stored per job:**
- `sus_reasons`: `str[]` — list of fired `rule_id` strings
- `rules_triggered`: JSONB — full objects with `rule_id`, `name`, `score`, `matched_keywords`, `matched_patterns`

Rules are binary per job: a rule either fires (matched ≥1 keyword or pattern, no exclusion hit) or doesn't. There is no partial score within a rule.

### 5. Poster/Employer Data (DB)

| Field | Type | Notes |
|-------|------|-------|
| `is_verified` | bool | Telegram verified |
| `is_premium` | bool | Telegram Premium subscriber |
| `entity_type` | str | user / chat / supergroup / channel |
| `total_jobs_posted` | int | Historical count |
| `average_sus_level` | float | Poster's average sus across all jobs |
| `trust_score` | float 0–1 | Computed reputation |
| `detected_patterns` | str[] | Recurring fraud patterns for this poster |

### 6. Raw Text (DB, `raw_content`)

The full original Telegram message text. Not vectorized yet, but available for embedding.

---

## What Can Be Added Easily

### A. Job Embeddings — High Value, Low Effort

The most impactful addition. Run `raw_content` through an embedding model → store as `pgvector` column in `jobs`.

**What this unlocks:**
- UMAP/t-SNE 2D scatter plot of all jobs — visually cluster by topic
- Semantic similarity search ("find jobs like this one")
- Detect near-duplicate postings from the same scam ring
- Anomaly detection: jobs that sit far from any cluster

**How to add:**
```python
# Already have: mxbai-embed-large:latest on server 25
# ollama pull mxbai-embed-large (already pulled)
response = requests.post("http://COMFY_HOST:11434/api/embeddings", json={
    "model": "mxbai-embed-large",
    "prompt": job.raw_content
})
embedding = response.json()["embedding"]  # 1024-dim float vector
```

Then store in Postgres with `pgvector`:
```sql
ALTER TABLE jobs ADD COLUMN embedding vector(1024);
CREATE INDEX ON jobs USING ivfflat (embedding vector_cosine_ops);
```

Backfill is one offline script. Production: embed at extraction time alongside LLM factor extraction.

**Cost:** `mxbai-embed-large` runs at ~50ms/item on server 25. 23k existing jobs = ~20 minutes to backfill.

---

### B. Factor Vector as Numeric Array

The 15 boolean factors + grammar_quality are already in `sus_factors` JSONB. They can be materialized as a numeric array for distance computations:

```python
# Convert sus_factors dict → fixed-length float vector
FACTOR_ORDER = [
    "has_specific_address", "has_company_or_org", "work_type_clear",
    "requires_experience", "has_phone_or_address", "has_salary_info",
    "grammar_quality",  # only continuous value
    "no_details_at_all", "only_dm_contact", "urgency_pressure",
    "targets_minors", "mentions_easy_money", "pay_work_mismatch",
    "bot_text_patterns", "suspicious_delivery", "critical_infrastructure",
]
vector = [float(factors.get(f, 0)) for f in FACTOR_ORDER]
```

No new LLM calls needed — reuse stored JSONB. Useful for:
- PCA/UMAP of fraud factor space
- Clustering jobs by fraud pattern type (e.g., "bot courier posts" vs "easy money posts")
- Heatmap of factor co-occurrence

---

### C. Salary Normalization

`salary_min` / `salary_max` extracted but in mixed currencies and periods. A normalized `salary_monthly_rub` field would let you:
- Plot salary vs sus_level scatter (do scams promise more?)
- Filter/color by pay bracket in visualizations

Simple offline computation: convert everything to RUB/month using fixed exchange rates.

---

### D. Text Features Without Embeddings

Extractable from `raw_content` with regex/heuristics — no LLM needed:

| Feature | How | Value |
|---------|-----|-------|
| `text_length` | `len(raw_content)` | Scam posts tend to be short |
| `has_phone_regex` | regex `\+7\|8-\d{3}` | Cross-check with LLM factor |
| `emoji_count` | count unicode emoji | Engagement bait signal |
| `cyrillic_latin_mix` | regex for lookalike chars | Bot text detection |
| `exclamation_count` | count `!` | Urgency indicator |
| `salary_mentioned_regex` | regex for `\d+[\s₽k]` | Cross-check with `has_salary_info` |
| `all_caps_ratio` | `sum(c.isupper()) / len` | Spammy formatting |

These are free and can be added as a batch migration.

---

## Recommended Stack for Visualization

For an offline exploration tool (Python notebook or Streamlit):

```
jobs table
  └── raw_content → mxbai-embed-large → 1024-dim embedding
  └── sus_factors → 16-dim factor vector
  └── sus_level, salary, work_type, source → color/filter axes

UMAP(n_components=2) on embeddings → 2D scatter
Color by: sus_category | work_type | source_name | mentions_easy_money
Size by: sus_level
Hover: title, company, raw_content snippet
```

Libraries: `umap-learn`, `plotly` (interactive), `pandas`, `sqlalchemy`.

---

## What's Missing / Gaps

| Gap | Impact | Fix |
|-----|--------|-----|
| No embeddings | Can't do semantic clustering | Add `pgvector` + embed at extraction time |
| `sus_factors` NULL for ~40% of jobs | Factor vector incomplete for older/failed jobs | Backfill or exclude from analysis |
| `work_type` is free text | Hard to group — "склад", "Склад", "warehouse" are different | Normalize to enum at extraction |
| No cluster labels | Can't train classifier on topic clusters | Label after first UMAP run |
| No time-series aggregation | Can't see fraud trend over time | Aggregate sus_level by week/source |
| `location` is free text | Geographic clustering broken | Normalize to city enum or geocode |

---

## Summary

| Layer | Status | Richness |
|-------|--------|----------|
| Structured metadata | ✅ Ready | Good |
| SUS score + category | ✅ Ready | Good |
| 15 fraud factors (JSONB) | ✅ Ready | High — directly clusterable |
| Poster reputation | ✅ Ready | Medium |
| Raw text | ✅ Stored | High potential, not exploited |
| Job embeddings | ❌ Missing | **Highest value add** |
| Normalized salary | ❌ Missing | Easy add |
| Text features (regex) | ❌ Missing | Easy add |

**The 15-factor vector is the fastest path to meaningful visualization** — no new infrastructure, just reshape the existing JSONB. Embeddings are the highest-value addition and unblock semantic clustering and similarity search.
