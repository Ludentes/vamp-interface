# Research: Embedding Model Selection for Russian Telegram Text

**Date:** 2026-04-06
**Sources:** 18 sources

---

## Executive Summary

The current model, `mxbai-embed-large`, is **English-only** and should be replaced immediately for Russian-language text. For Ollama-local deployment on ~27k informal Russian Telegram posts, the best available option is **`qwen3-embedding:0.6b`** (MTEB Multilingual rank #1 at the 8B scale; the 0.6B variant scores 64.33 and fits on consumer hardware). If inference speed or VRAM is a constraint, **`bge-m3`** is the proven multilingual workhorse with an official Ollama tag, ruMTEB score of 60.8, and strong Russian retrieval performance. The instruct variant of multilingual-e5-large achieves the best ruMTEB clustering score among sub-1B models (63.3) but is only available via a community Ollama namespace, adding deployment risk.

---

## Key Findings

### mxbai-embed-large is English-only

The HuggingFace model card for `mixedbread-ai/mxbai-embed-large-v1` explicitly lists "English" as the sole supported language [1]. The model achieves SOTA for BERT-large sized models on the English MTEB (64.68 average across 56 datasets, STS 85.00) [1], but this benchmark is English-only. It was not submitted to ruMTEB or MMTEB in any of the sources reviewed. Using this model on Cyrillic text produces embeddings that likely collapse Russian vocabulary into an out-of-distribution space — the model has no Russian tokens in its vocabulary. This is a fundamental issue, not a performance degradation: semantically similar Russian posts will not cluster together.

Similarly, `nomic-embed-text` v1.5 (the version available in the Ollama official library as `nomic-embed-text`) is English-only. A Nomic AI team member confirmed in a HuggingFace discussion: "It's only English...for now" [3]. The v2-moe variant supports ~100 languages and is available via `nomic-embed-text-v2-moe` on Ollama, but the default `nomic-embed-text` pull is the English-only v1/v1.5.

### ruMTEB: the benchmark that matters

ruMTEB provides 23 Russian-language datasets across 7 task categories: Classification (9), Clustering (3), MultiLabel Classification (2), Pair Classification (1), Reranking (2), Retrieval (3), STS (3) [5]. It is the authoritative Russian-specific embedding benchmark and was presented at NAACL 2025.

Scores from the ruMTEB paper [5] for models evaluated on Russian clustering tasks:

| Model | ruMTEB avg | Clustering score |
|---|---|---|
| E5-mistral-7b-instruct | 64.9 | 64.0 |
| mE5large-instruct | 64.7 | 63.3 |
| BGE-M3 | 60.8 | 52.2 |
| ru-en-RoSBERTa | 60.4 | 56.1 |
| mE5large | ~57 (est.) | 52.5 |

`mxbai-embed-large`, `nomic-embed-text`, `LaBSE`, and `paraphrase-multilingual-mpnet` were **not evaluated** in the ruMTEB paper — no scores available.

GigaEmbeddings (Sber, 2025) achieves 69.1 ruMTEB average and ranks first on both MTEB and CMTEB [7], but is **not available on Ollama** as of this writing.

### Qwen3-embedding: best available on Ollama

Qwen3-Embedding was released in June 2025 and ranks #1 on the MTEB Multilingual leaderboard [8][9]:

| Variant | MTEB Multilingual | Embedding dim | Ollama tag |
|---|---|---|---|
| 0.6B | 64.33 | 1024 | `qwen3-embedding:0.6b` |
| 4B | 69.45 | 2560 | `qwen3-embedding:4b` |
| 8B | 70.58 | 4096 | `qwen3-embedding:8b` |

All variants support MRL (Matryoshka Representation Learning), meaning dimensions can be truncated (e.g., to 512 or 256) without retraining. Russian is explicitly listed as a supported language in the Qwen3 GitHub README [8]. The 0.6B variant at 1024-d is a practical choice for 27k documents: it outperforms multilingual-e5-large on MTEB Multilingual (64.33 vs. ~58.7 for mE5large) while running on a single GPU. The 0.6B model is in the official `ollama.com/library/qwen3-embedding` namespace [10].

### BGE-M3: proven multilingual model with official Ollama support

BGE-M3 from BAAI is available as `bge-m3` in the official Ollama library [11]. It supports 100+ languages, handles inputs up to 8192 tokens, and achieves a ruMTEB score of 60.8 [5]. On MTEB English it scores 64.2 average [12]. It produces **1024-dimensional** embeddings. On Mr. TyDi Russian retrieval, it belongs to the top-performing sub-1B models [5][12]. Its architecture (XLM-RoBERTa-based) was specifically pre-trained on multilingual corpora exceeding 1 trillion tokens, which gives it robust Cyrillic coverage including informal register, abbreviations, and mixed-script text [12].

BGE-M3's 8192-token context window is an advantage for longer posts or batch encoding. For a 27k-document clustering task with short Telegram posts (~50-200 tokens each), this headroom does not translate to a measurable quality difference, but it avoids truncation edge cases.

### multilingual-e5 family: strong but community Ollama namespace only

The `multilingual-e5-large` and `multilingual-e5-large-instruct` models (Microsoft/intfloat) achieve strong ruMTEB scores — the instruct variant leads non-LLM models at 63.3 clustering [5]. Embedding dimension is 1024. On Mr. TyDi Russian retrieval MRR@10, mE5large scores 65.8 (vs. mE5base at 62.7) [4].

However, neither is in the official Ollama library namespace. They are available via `jeffh/intfloat-multilingual-e5-large` (community namespace) [14]. Community-namespaced Ollama models carry higher deployment risk: no guaranteed maintenance, potential removal, and no official quantization guarantees. The instruct variant requires a task prefix (e.g., `"query: "` / `"passage: "`) — for clustering where all texts are symmetric, you would use `"passage: "` for all inputs, which is slightly non-standard usage.

### paraphrase-multilingual-mpnet-base-v2: usable but outdated

Available as `paraphrase-multilingual` in the official Ollama library [15]. Supports 50 languages including Russian, produces 768-d embeddings, 278M parameters. The critical limitation: **128-token maximum sequence length**. Telegram posts with emoji, URLs, and text can easily exceed this. Text is silently truncated, degrading embedding quality for longer posts. It was state-of-the-art circa 2021 but has been comprehensively outperformed by newer models.

### Dimensionality tradeoffs for clustering at ~27k documents

For the visualization + clustering use case (UMAP → HDBSCAN), the embedding dimension primarily affects the quality of the initial semantic space, not the clustering step itself. UMAP reduces to 2D for visualization regardless of input dimension [16]. At 27k documents, even 1024-d embeddings are computationally trivial for UMAP (it scales well with dataset size and is commonly applied to 100k+ documents [16]).

The practical tradeoff at this scale is: higher-dimensional embeddings from better models (1024-d Qwen3-0.6B, 1024-d BGE-M3) will produce better cluster separation than lower-dimensional embeddings from weaker models. There is no benefit to choosing a 512-d model over a 1024-d model for this corpus size — the curse of dimensionality does not manifest until you reach very high dimensions (thousands) with limited data. Qwen3's MRL feature is useful if you want to experiment with reduced dimensions post-hoc without re-embedding.

For informal Telegram text specifically: abbreviations like "ЗП", "р/ч", emoji, and Cyrillic-Latin substitutions (e.g., "c" for "с") are present in training data for any model trained on large web corpora. BGE-M3 and Qwen3-Embedding are both trained on web-scale multilingual data that includes Russian social media text. Paraphrase-multilingual was trained on cleaner parallel corpora and may handle informal register less robustly — though no specific benchmark exists for this claim (single-source concern).

---

## Comparison Table

| Model | Ollama tag | Dims | Russian MTEB | Notes |
|---|---|---|---|---|
| **qwen3-embedding:0.6b** | `qwen3-embedding:0.6b` (official) | 1024 (MRL) | MTEB Multilingual 64.33 [9] | Best overall; 0.6B fits on consumer GPU |
| **bge-m3** | `bge-m3` (official) | 1024 | ruMTEB 60.8 [5] | Proven, official, 8192-token context |
| multilingual-e5-large-instruct | `jeffh/intfloat-multilingual-e5-large` (community) | 1024 | ruMTEB clustering 63.3 [5] | Instruct variant requires task prefix; community namespace |
| multilingual-e5-large | `jeffh/intfloat-multilingual-e5-large` (community) | 1024 | ruMTEB clustering 52.5 [5] | Community namespace only |
| nomic-embed-text-v2-moe | `nomic-embed-text-v2-moe` (official) | 768 (MRL to 256) | MIRACL multilingual 65.80 [2] | ~100 langs; no ruMTEB score available |
| paraphrase-multilingual | `paraphrase-multilingual` (official) | 768 | Not on ruMTEB; outdated | **128-token max** — truncates longer posts |
| **mxbai-embed-large** | `mxbai-embed-large` (official) | 1024 | **English-only — DO NOT USE** [1] | Currently installed; unsuitable for Russian |
| nomic-embed-text (v1.5) | `nomic-embed-text` (official) | 768 | **English-only** [3] | Default Ollama tag is English-only |
| GigaEmbeddings | Not available | 1024 | ruMTEB 69.1 [7] | Best Russian model; not on Ollama |

---

## Recommendation

**Replace `mxbai-embed-large` with `qwen3-embedding:0.6b`.**

Pull command: `ollama pull qwen3-embedding:0.6b`

Rationale:
- Highest MTEB Multilingual score (64.33) of any model in the official Ollama library at this parameter count
- Official Ollama namespace (not community) — stable, maintained
- 1024-d with MRL support for dimension experimentation
- Explicit Russian language support confirmed
- Fits on the same hardware as mxbai-embed-large

If you want the most comparable deployment to the current setup with minimal risk, **`bge-m3`** is the conservative alternative — official namespace, proven on Russian, ruMTEB 60.8, well-documented community usage. It scores slightly lower than Qwen3-0.6B on MTEB Multilingual but has a longer track record and more community validation for Russian specifically.

To validate before full re-embedding: embed a sample of 100–200 known-similar posts (same job category) and 100–200 known-different posts with each model, then compute intra-cluster vs. inter-cluster cosine similarity distributions. This takes ~5 minutes and will empirically confirm cluster separation quality before committing to re-embedding all 27k posts.

---

## Open Questions

1. **Qwen3-embedding Russian-specific ruMTEB score**: The Qwen3 paper provides MTEB Multilingual aggregate and Chinese/English breakdowns, but not per-language scores for Russian on ruMTEB tasks [8][9]. This is the main gap — the model may perform even better than implied by the aggregate.

2. **nomic-embed-text-v2-moe on Ollama**: The Ollama page exists (`nomic-embed-text-v2-moe`) but access was denied during research. Its MIRACL multilingual score of 65.80 is promising but MIRACL measures retrieval, not clustering/STS. No ruMTEB score available.

3. **Informal register robustness**: No benchmark specifically tests Telegram-style informal Russian (abbreviations, emoji, Cyrillic-Latin substitutions). The recommendation is based on general multilingual performance. Empirical sampling test (see Recommendation) is essential.

4. **GigaEmbeddings Ollama availability**: Best ruMTEB model (69.1) from Sber, open-sourced as of late 2025 [7]. Worth checking if a community Ollama packaging has appeared after this research date.

---

## Sources

[1] mixedbread-ai. "mxbai-embed-large-v1". https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1 (Retrieved: 2026-04-06)

[2] nomic-ai. "nomic-embed-text-v2-moe". https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe (Retrieved: 2026-04-06)

[3] nomic-ai. "nomic-embed-text-v1.5 — Supported languages? (Discussion)". https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/discussions/9 (Retrieved: 2026-04-06)

[4] intfloat/Microsoft. "multilingual-e5-large". https://huggingface.co/intfloat/multilingual-e5-large (Retrieved: 2026-04-06)

[5] Snegirev et al. "The Russian-focused embedders' exploration: ruMTEB benchmark and Russian embedding model design". NAACL 2025. https://arxiv.org/html/2408.12503v1 (Retrieved: 2026-04-06)

[6] embeddings-benchmark. "MMTEB: Massive Multilingual Text Embedding Benchmark". arXiv:2502.13595. https://arxiv.org/abs/2502.13595 (Retrieved: 2026-04-06)

[7] Sber. "GigaEmbeddings — Efficient Russian Language Embedding Model". ACL Anthology BSNLP 2025. https://aclanthology.org/2025.bsnlp-1.3/ (Retrieved: 2026-04-06)

[8] QwenLM. "Qwen3-Embedding GitHub README". https://github.com/QwenLM/Qwen3-Embedding (Retrieved: 2026-04-06)

[9] Qwen. "Qwen3-Embedding-8B model card". https://huggingface.co/Qwen/Qwen3-Embedding-8B (Retrieved: 2026-04-06)

[10] Ollama. "qwen3-embedding library page". https://ollama.com/library/qwen3-embedding (Retrieved: 2026-04-06)

[11] Ollama. "bge-m3 library page". https://ollama.com/library/bge-m3 (Retrieved: 2026-04-06)

[12] johal.in. "BGE M3 Multilingual: Massive Embeddings for Global RAG Systems 2025". https://johal.in/bge-m3-multilingual-massive-embeddings-for-global-rag-systems-2025-3/ (Retrieved: 2026-04-06)

[13] ollama.com. "multilingual-e5-large community model (issue)". https://github.com/ollama/ollama/issues/3606 (Retrieved: 2026-04-06)

[14] jeffh/Ollama. "intfloat-multilingual-e5-large". https://ollama.com/jeffh/intfloat-multilingual-e5-large:f32 (Retrieved: 2026-04-06)

[15] Ollama. "paraphrase-multilingual library page". https://ollama.com/library/paraphrase-multilingual (Retrieved: 2026-04-06)

[16] umap-learn. "Using UMAP for Clustering". https://umap-learn.readthedocs.io/en/latest/clustering.html (Retrieved: 2026-04-06)

[17] sentence-transformers. "paraphrase-multilingual-mpnet-base-v2". https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (Retrieved: 2026-04-06)

[18] BAAI. "bge-m3". https://huggingface.co/BAAI/bge-m3 (Retrieved: 2026-04-06)
