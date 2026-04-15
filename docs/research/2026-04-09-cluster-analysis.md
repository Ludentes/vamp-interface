# Cluster Analysis Report — Telejobs Corpus

**Date:** 2026-04-09
**Dataset:** `output/full_layout.parquet`
**Corpus:** 23,777 job postings, cutoff 2026-02-15
**Layout:** PaCMAP on 1024-d mxbai-embed-large embeddings, `n_neighbors=15`, seed=42
**Clustering:** HDBSCAN on PaCMAP 2D coordinates
  - Fine clusters: 195 (+ noise)
  - Coarse clusters: 42

---

## 1. Corpus Overview

| Metric | Value |
|--------|-------|
| Total jobs | 23,777 |
| Fine clusters | 195 |
| Coarse clusters | 42 |
| Noise (unassigned) | 2,757 (11.6%) |
| Sources (channels) | 20 tracked |
| Date range | 2026-02-15 → 2026-04-07 |

**Sus level distribution:**

| Band | Count | % |
|------|-------|---|
| 0–9 (safe) | 3,286 | 13.8% |
| 10–19 | 929 | 3.9% |
| 20–29 | 8,552 | 36.0% |
| 30–39 | 1,372 | 5.8% |
| 40–49 | 315 | 1.3% |
| 50–59 | 1,966 | 8.3% |
| 60–69 | 705 | 3.0% |
| 70–79 | 306 | 1.3% |
| 80–89 | 436 | 1.8% |
| 90–99 | 1,386 | 5.8% |
| 100 | 4,524 | 19.0% |

**By category:**

| Category | Count | % |
|----------|-------|---|
| low (sus 0–30) | 9,889 | 41.6% |
| critical (sus 90–100) | 6,335 | 26.6% |
| safe (sus 0–15) | 4,331 | 18.2% |
| medium (sus 31–69) | 2,336 | 9.8% |
| high (sus 70–89) | 886 | 3.7% |

---

## 2. Bimodal Structure

The corpus is strongly bimodal. Safe and fraud are the two poles; the ambiguous band is thin.

| Zone | sus range | Count | % |
|------|-----------|-------|---|
| Safe | 0–30 | 12,835 | **54.0%** |
| Ambiguous | 31–69 | 4,290 | **18.0%** |
| Fraud | 70–100 | 6,652 | **28.0%** |

The bimodality is structural, not an artifact of the labeling model — the PaCMAP layout shows two distinct spatial regions, confirming that safe and fraud jobs use genuinely different language.

---

## 3. Spatial Separation

PaCMAP places fraud and safe posts in distinct regions of the 2D layout.

| Zone | x mean | x std | x median |
|------|--------|-------|---------|
| Safe (sus ≤ 30) | **0.578** | 0.164 | 0.561 |
| Ambiguous (31–69) | 0.373 | 0.184 | 0.441 |
| Fraud (sus ≥ 70) | **0.370** | 0.160 | 0.373 |

**Key finding:** Safe jobs cluster at x ≈ 0.57 (right side of the map); fraud jobs cluster at x ≈ 0.37 (left side). The X-axis encodes the safe/fraud axis with a 0.21-unit mean separation — visible as a spatial divide when coloring by sus_level.

The Y-axis encodes work-type language (physical labor at bottom, remote/light at top) but does not separate fraud from safe cleanly.

---

## 4. Work-Type Distribution

| Work type | Count | % of corpus |
|-----------|-------|-------------|
| другое (other) | 5,894 | 24.8% |
| (unlabeled) | 5,181 | 21.8% |
| доставка (delivery) | 2,829 | 11.9% |
| стройка (construction) | 1,982 | 8.3% |
| уборка (cleaning) | 1,859 | 7.8% |
| склад (warehouse) | 1,626 | 6.8% |
| офис (office) | 1,480 | 6.2% |
| удалёнка (remote) | 1,071 | 4.5% |
| погрузка (loading) | 959 | 4.0% |
| торговля (retail) | 562 | 2.4% |
| общепит (food service) | 332 | 1.4% |

**Delivery (доставка) is the dominant fraud vector** — it appears as the top work type in 7 of the 12 highest-fraud coarse clusters (C0, C3, C8, C9, C11, C14, C16).

---

## 5. Coarse Cluster Profiles

42 coarse clusters. Listed by mean sus level — pure fraud at top, pure safe at bottom.

| Cluster | N | Sus mean | % Fraud | % Safe | Top work type | Character |
|---------|---|----------|---------|--------|---------------|-----------|
| C11 | 152 | **100.0** | 100% | 0% | доставка | Pure fraud — delivery |
| C12 | 186 | **95.3** | 94.6% | 4.3% | другое | Near-pure fraud — misc |
| C8 | 161 | **94.8** | 99.4% | 0% | удалёнка | Pure fraud — remote work |
| C9 | 186 | **93.7** | 97.3% | 2.7% | доставка | Near-pure fraud — delivery |
| C14 | 163 | **88.3** | 88.3% | 11.7% | доставка | High fraud — delivery |
| C16 | 593 | 77.7 | 57.0% | 6.2% | доставка | Mixed-high fraud — delivery |
| C3 | 167 | 73.6 | 67.1% | 32.9% | доставка | Mixed fraud — delivery |
| C15 | 211 | 72.7 | 64.9% | 26.1% | другое | Mixed fraud — misc |
| C25 | 152 | 71.2 | 70.4% | 23.7% | другое | Mixed fraud — misc |
| C2 | 415 | 69.5 | 61.2% | 28.7% | другое | Mixed fraud |
| C10 | 430 | 67.9 | 60.2% | 26.3% | (mixed) | Mixed fraud |
| C0 | 1,197 | 63.0 | 55.3% | 36.9% | доставка | **Largest fraud cluster** — delivery |
| ... | | | | | | |
| C4 | 384 | 25.0 | **0%** | 100% | уборка | Pure safe — cleaning |
| C7 | 373 | 25.0 | **0%** | 100% | уборка | Pure safe — cleaning |
| C26 | 198 | 23.4 | **0%** | 100% | (mixed) | Pure safe |
| C28 | 158 | 22.5 | **0%** | 100% | (mixed) | Pure safe |

**C0 is the most strategically important cluster** — the largest fraud-dominant cluster at 1,197 jobs. Delivery-sector fraud is concentrated here. Mean sus 63, 55% fraud rate — high enough to be visually flaggable.

The large heterogeneous cluster C20 (9,624 jobs, 40% mean sus) is effectively the "corpus bulk" — containing the mixed background of mostly-safe posts from the main channel.

---

## 6. Fine Cluster Purity

195 fine clusters (n ≥ 20 subset: 195 → 130 filtered).

| Purity | Count | Description |
|--------|-------|-------------|
| Very pure (sus_std < 15) | **121** | Tightly homogeneous sus level |
| Mixed (sus_std 15–30) | 38 | Internally varied |
| Impure (sus_std > 30) | 36 | High within-cluster variance |

**93% of fine clusters are at least somewhat cohesive.** The 121 very-pure clusters are the game's best content — a face from a cluster with sus_std < 15 has a "correct" uncanny level that is consistent with its neighbors.

**Top pure fraud fine clusters** (sus_mean = 100.0, all-critical):

| Fine cluster | N | Top work type | Coarse |
|-------------|---|---------------|--------|
| 44 | 145 | доставка | noise |
| 79 | 152 | доставка | C11 |
| 89 | 127 | доставка | C16 |
| 75 | 186 | другое | C12 |
| 45 | 186 | доставка | C9 |

These are the densest pure-fraud zones in the map. All delivery or misc category.

---

## 7. Source Channel Analysis

| Channel | N | Sus mean | % Critical (sus≥90) | Character |
|---------|---|----------|---------------------|-----------|
| @rabota_v_permi_59 | 15,689 | 48.7 | 25.9% | Main channel — mixed |
| @Kirov_Avto_rynok | 2,308 | 52.0 | 27.3% | Car/auto adjacent — elevated |
| **@shabashka2** | 2,127 | **65.4** | **48.9%** | **High-fraud channel** |
| @vsem_podryad | 910 | 12.3 | 0.2% | Clean channel — mostly legitimate |
| @Rabota_Kirovx | 377 | 12.0 | 1.1% | Clean channel |
| @PERM_PA6OTA | 365 | 23.5 | 8.2% | Mostly safe |
| @kirov_rabotab | 362 | 12.8 | 1.9% | Clean channel |
| @Udalennaya_rabotai | 100 | **0.4** | 0% | **Cleanest channel** — near-zero fraud |

### @shabashka2 — Fraud channel deep dive

This channel has the highest fraud concentration of any high-volume source (2,127 jobs, mean sus 65.4, 48.9% critical). Notable structure:

- Work types: mostly "другое" (969) and unlabeled (416) — obscured category
- Sus distribution is **trimodal**: large clean cluster at sus=20 (439 jobs), large fraud cluster at sus=90+ (1,040+ jobs), small middle band
- The clean-looking sus=20 posts likely serve as cover — legitimate-looking postings mixed with high-fraud content in the same channel

This is the most structurally suspicious channel in the corpus.

---

## 8. Implications for Scam Guessr

### Game content quality

The bimodal distribution (54% safe / 28% fraud) is well-suited for the game. Players will see a genuine mix, not a fraud-dominated set.

The 121 pure fine clusters mean most faces have consistent uncanny levels within their spatial neighborhood — the map's visual continuity property holds.

### Face generation coverage gap

The current test set (543 faces) covers the cohort-sampled subset. The full corpus generation (24k jobs running now in telejobs) will cover most of the bimodal range. The thin ambiguous band (sus 31–69, only 18% of corpus) is underrepresented in pure clusters — those faces will have the most varied/inconsistent generation results.

### High-value game clusters

For a "hard mode" or expert session, faces drawn from the **pure fraud fine clusters** (44, 79, 89, 75, 45) and the **pure safe clusters** (C4, C7, C26, C28) would maximize calibration challenge — the faces look wrong (or right) with high consistency.

### Channel as a fraud signal

@shabashka2's trimodal sus distribution is the strongest single-channel fraud signal. If channel metadata is ever surfaced in the game (e.g. "source cluster" hint), this channel would be a reliable hard-difficulty indicator.

---

## 9. Data Quality Notes

- **21.8% of jobs have no work_type** (NULL) — labeled as "unknown" in cohort assignment
- **cluster = -1 (noise)** accounts for 2,757 jobs (11.6%) — these are semantic outliers that didn't fit any HDBSCAN cluster
- The sus=20–29 band is by far the most populous (36% of corpus) — this reflects the model's tendency to score ambiguously-worded but possibly legitimate posts in this band
- The clean spike at sus=100 (4,524 jobs, 19%) represents a confident critical-fraud zone — these posts are maximally unambiguous to the model
