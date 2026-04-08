# LoRA Usage Notes — Uncanny Valley Generation

**Date:** 2026-04-08  
**Purpose:** Drive uncanny valley effect for high-sus faces in vamp-interface v5+

---

## Cursed Flux LoRA

**File:** `Cursed_LoRA_Flux.safetensors`  
**Source:** https://civitai.com/models/655938  
**Base model:** Flux.1-Dev  
**Size:** 19 MB

**Trigger words / effective keywords:**
- `cursed` — primary trigger
- `scary looking`, `scary appearance`
- `sharp teeth`, `many teeth`
- `undead figure`
- (gore keywords like `cuts and wounds` — avoid for our use case)

**Recommended weight range:** 0.5–1.5  
- Below 0.5: subtle distortion, wrong bone structure hints
- 0.5–0.8: uncanny geometry, wrong proportions — **our sweet spot**
- Above 1.0: full monster/zombie territory — too obvious

**Our mapping:** `weight = (sus / 100) * 0.6` → 0.0 at sus=0, 0.6 at sus=100

**Notes:**
- Tested on ForgeUI with Q8_0.gguf (better quality than Q4)
- No specific steps/sampler requirement; works with euler/simple
- Stack with other LoRAs — combine with Eerie for compound effect
- The wrongness is structural (bone, teeth, jaw) rather than atmospheric

---

## Eerie Horror Portraits

**File:** `Eerie_horror_portraits.safetensors`  
**Source:** https://civitai.com/models/829793  
**Base model:** Flux.1-Dev  
**Size:** 19 MB

**Trigger words / effective keywords (no hard trigger, use descriptors):**
- `eerie`, `ominous`, `haunting`
- `hollow`, `ghostly`, `pale`
- `high contrast`, `dramatic lighting`
- `dark`, `macabre`, `gothic`
- `skeletal` (for extreme cases)

**Recommended weight range:** 0.3–1.0  
- Below 0.3: darkens mood, tightens expression — minimal
- 0.3–0.6: hollow eyes, uncomfortable expression — **our sweet spot**
- Above 0.8: heavily stylised dark portrait, loses photorealism

**Our mapping:** `weight = (sus / 100) * 0.5` → 0.0 at sus=0, 0.5 at sus=100

**Notes:**
- Primarily affects expression and lighting feel, not geometry
- Works well as a secondary layer on top of Cursed
- Adds the "hollow presence behind the eyes" effect we want at high sus
- No sampler or steps requirements documented

---

## Combined usage (v5 generation)

Stack both LoRAs in the Flux workflow via two `LoraLoader` nodes chained:

```
UNETLoader → LoraLoader(Cursed, weight=cursed_w) → LoraLoader(Eerie, weight=eerie_w) → KSampler
```

Sus-scaled weights:
```python
cursed_w = (sus / 100) * 0.6   # 0.0 → 0.60
eerie_w  = (sus / 100) * 0.5   # 0.0 → 0.50
```

Prompt additions for high sus (appended to descriptor):
- `cursed` (activates Cursed LoRA)
- `eerie`, `hollow`, `haunting` (activates Eerie LoRA)

At sus=0: both weights ~0, prompts neutral → clean portrait
At sus=100: Cursed=0.6 + Eerie=0.5 → wrong geometry + hollow expression

**Guidance scale:** scale from 3.5 (sus=0) to 6.0 (sus=100) for stronger prompt adherence at high sus.

---

## What did NOT work

- **Uncanny Valley LoRA (civitai 919502):** turned out to be a retro 1950s kitsch humor model, not horror. Avoided.
- **Pure prompting without LoRA:** Flux's realism prior overrides geometric defect descriptions. The uncanny terms land at end of prompt and get downweighted. LoRA is required.
- **Denoising alone (0.85):** At high denoise, Flux ignores the anchor and generates a clean portrait driven by the prompt. Without LoRA the prompt still produces a normal face.
