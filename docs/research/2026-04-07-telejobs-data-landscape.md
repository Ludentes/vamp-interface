# telejobs Data Landscape Analysis

**Date**: 2026-04-07  
**Purpose**: Understand what structured information is available in the telejobs corpus for improving the PCA→face mapping.

---

## Corpus Overview

| Metric | Value |
|--------|-------|
| Total jobs | 27,212 |
| With embedding | 27,212 (100%, backfill complete) |
| Extraction completed | 25,768 (94.7%) |
| Extraction pending | 1,444 (5.3%) |

---

## Sus Level Distribution

```
sus   0–9 :  5,714  (21%)   safe/low baseline
sus  10–19:  1,065  ( 4%)
sus  20–29:  8,650  (32%)   ← large cluster of "probably legit" jobs
sus  30–39:  1,445  ( 5%)
sus  40–49:    490  ( 2%)   ← thin middle: few borderline cases
sus  50–59:  2,135  ( 8%)
sus  60–69:    766  ( 3%)
sus  70–79:    330  ( 1%)
sus  80–89:    482  ( 2%)
sus  90–99:  1,436  ( 5%)
sus 100   :  4,699  (17%)   ← large cluster of confirmed critical fraud
```

**Bimodal distribution**: corpus splits into two modes — sus 20–29 (legit-ish) and sus 100 (critical). Thin middle band 40–79. This is good for face generation contrast; bad for uncanny valley gradients in the intermediate range.

---

## Sus Category Distribution

| Category | Count | Sus Range |
|----------|-------|-----------|
| low      | 10,058 | ~0–29    |
| safe     |  6,928 | ~0–15    |
| critical |  6,590 | ~80–100  |
| medium   |  2,678 | ~30–60   |
| high     |    958 | ~60–80   |

---

## Work Type × Fraud (Critical Finding)

| Work Type | Avg Sus | Count | Interpretation |
|-----------|---------|-------|----------------|
| доставка (delivery) | **83.1** | 2,834 | Almost universally scam |
| удалёнка (remote)   | **64.9** | 1,080 | High fraud rate |
| другое (other)      | **61.6** | 5,950 | Mixed |
| склад (warehouse)   | 36.2 | 1,629 | Mostly legit |
| погрузка (loading)  | 34.4 |   966 | Mostly legit |
| офис (office)       | 31.5 | 1,491 | Mostly legit |
| торговля (trade)    | 27.3 |   566 | Mostly legit |
| стройка (construction) | 26.2 | 2,015 | Mostly legit |
| уборка (cleaning)   | 26.1 | 1,864 | Mostly legit |

**Key implication**: `доставка` (courier/delivery) is a scam-dominated category (avg_sus=83).
Our "courier_legit" cluster (sus_max=30, n=2,834 total) draws from the tail — only ~3% of delivery posts are non-fraud. This tail may have embedding characteristics closer to `scam_critical` than to `warehouse_legit`.

---

## Field Fill Rates

| Field | Fill Rate | Notes |
|-------|-----------|-------|
| description | 100% | `raw_content` always present |
| sus_level | 100% | |
| sus_category | 100% | |
| sus_factors | 87% | 16-key JSONB |
| salary_min | 76% | |
| salary_currency | 77% | mostly RUB |
| title | 69% | |
| work_type | 68% | |
| contact_method | 58% | |
| required_skills | 44% | array |
| payout | 33% | |
| salary_max | 31% | |
| location | 16% | sparse |
| job_type | 18% | |
| experience_level | 14% | noisy (multilingual: "без опыта", "entry level", "junior") |
| company | 8% | rare — another fraud signal |
| positions_available | 3% | |

**Sparsity concern**: location, company, experience_level are too sparse for reliable PCA axes.
**Rich fields**: sus_factors (87%), salary_min (76%), work_type (68%), required_skills (44%).

---

## Sus Factors: 16 Binary/Float Indicators

All 15 boolean factors + 1 float (`grammar_quality: 0.2–1.0`). All have ~6% null rate.

### Ranked by predictive power (delta = avg_sus if_true − avg_sus if_false)

| Factor | Delta | Sus if True | Sus if False | N True |
|--------|-------|-------------|--------------|--------|
| `pay_work_mismatch` | **+57.6** | 98.8 | 41.2 | 2,964 |
| `mentions_easy_money` | **+61.7** | 97.8 | 36.1 | 4,601 |
| `suspicious_delivery` | **+52.0** | 95.7 | 43.7 | 2,198 |
| `bot_text_patterns` | **+42.2** | 89.3 | 47.1 | 887 |
| `no_details_at_all` | **+41.3** | 80.9 | 39.5 | 5,010 |
| `only_dm_contact` | **+37.4** | 53.1 | 15.7 | 19,719 |
| `urgency_pressure` | **+13.3** | 60.3 | 47.0 | 3,094 |
| `targets_minors` | **+18.2** | 66.3 | 48.1 | 858 |
| `critical_infrastructure` | **-1.6** | 47.2 | 48.8 | 176 |
| `has_salary_info` | **-1.0** | 48.7 | 49.6 | 18,678 |
| `requires_experience` | **-30.6** | 20.9 | 51.5 | 1,980 |
| `has_company_or_org` | **-33.5** | 17.8 | 51.3 | 1,651 |
| `has_phone_or_address` | **-33.7** | 17.9 | 51.6 | 1,844 |
| `work_type_clear` | **-36.5** | 37.8 | 74.2 | 15,523 |
| `has_specific_address` | **-41.0** | 10.2 | 51.2 | 1,307 |

### Grammar Quality (continuous)

| Quality | Avg Sus | N |
|---------|---------|---|
| low (<0.4) | 87.3 | 381 |
| mid (0.4–0.7) | 81.0 | 1,764 |
| high (>0.7) | 45.3 | 20,132 |

`grammar_quality` is the only continuous sus_factor. It correlates with sus_level and could directly drive the uncanny valley gradient as a face "texture quality" axis.

---

## Salary × Fraud

| Sus Category | Avg Salary Min (RUB) | N |
|-------------|---------------------|---|
| safe | 42,155 | 3,577 |
| critical | 17,665 | 4,130 |
| medium | 17,640 | 2,267 |
| high | 17,367 | 403 |
| low | 8,605 | 8,600 |

Safe/professional jobs advertise highest salaries. Critical fraud jobs sit around market average (17k RUB) — they're not "too good to be true" on salary alone; `mentions_easy_money` (n=4,601) is a more targeted signal.

---

## Implications for Face Generation Mapping

### Better axes for the identity channel

| PCA-derived axis | What it probably encodes | Face mapping |
|------------------|--------------------------|--------------|
| PC1 (+): physical/manual | склад, погрузка, стройка | Older, weathered, stocky |
| PC1 (−): office/knowledge | офис, удалёнка | Younger, slim, professional |
| PC2 (+): formal posting style | structured, has_company | Androgynous, groomed, business |
| PC2 (−): informal | casual postings | Masculine, practical dress |

### Better axes for the expression channel

Instead of only `sus_level`, consider using specific sus_factors directly:

| Factor | Face expression cue |
|--------|---------------------|
| `grammar_quality` (float, continuous) | Uncanny proportions — low grammar → more deformed symmetry |
| `mentions_easy_money` | "Too-wide smile, hollow excitement" |
| `pay_work_mismatch` | "Performative enthusiasm, overclaiming" |
| `no_details_at_all` | "Vague, evasive gaze, non-specific" |
| `urgency_pressure` | "Tense, slightly aggressive" |
| `has_specific_address` (protective) | "Open, grounded expression" |

### Cluster suggestions for Phase 3+

Current 4 clusters don't cover the full fraud spectrum. Recommended expansion:

| Cluster | Filter | Expected sus | Purpose |
|---------|--------|-------------|---------|
| warehouse_legit | work_type=склад, sus≤30 | 20–30 | Physical legit baseline |
| office_legit | work_type=офис, sus≤30 | 6–24 | White-collar legit baseline |
| courier_legit | work_type=доставка, sus≤30 | 0–27 | Rare legit couriers |
| courier_scam | work_type=доставка, sus≥80 | 80–100 | Dominant courier fraud |
| remote_scam | work_type=удалёнка, sus≥60 | 60–100 | Remote work fraud |
| easy_money | mentions_easy_money=true, sus≥90 | 90–100 | "Get rich quick" scams |

The `courier_legit` vs `courier_scam` pairing would be the strongest test of the identity+expression channels — same work type, but one cluster is predominantly fraudulent.

---

## Data Quality Notes

- `experience_level`: noisy multilingual (Russian + English, inconsistent capitalization). Not usable as-is.
- `location`: 16% fill rate — too sparse for spatial analysis.
- `company`: 8% fill — rarity itself is a fraud signal (legitimate companies name themselves).
- `salary_min` outliers: some entries have salary >1M RUB (likely data errors); filter to <500k for analysis.
- `only_dm_contact`: 88.5% of all jobs use DM-only contact — this field has limited discriminating power because it's nearly universal.
