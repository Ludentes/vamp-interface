# Citation Check — Perception Before Training

Source file: `docs/blog/2026-04-20-perception-before-training.md`
Checked: 2026-04-20

## Summary

- **6 FAITHFUL**, **2 GENEROUS**, **0 MISATTRIBUTED**, **0 OVERREACH**, **0 UNVERIFIABLE** across the eight numbered citations.
- The author's empty-skill-landscape claim is plausible but only spot-checked (I did not exhaustively read every awesome-list).
- All three tool pointers (AEPsych, PsychoPy, jspsych-psychophysics) verify cleanly.

**Top three fixes before publishing:**
1. The "21-word solution" year in the footer reads `Simmons-Nelson-Simonsohn (2012)`. That is correct for the SSRN note that coined the 21 words, but the body also invokes the "False-Positive Psychology" argument (2011, Psychological Science) as if it were the same paper. If the blog is cited as methodology grounding, cite *both* to avoid giving readers the impression the 21-word statement stands alone rather than on the 2011 paper's evidence base.
2. Tighten the Smith & Little (2018) framing. The paper defends small-N as a class, using the individual participant as replication unit across *many* trials; it does *not* endorse "N=3–5" as a fixed range. The author's phrasing ("license for N=3–5 designs") is defensible given standard vision-science practice but is stronger than the paper itself asserts. Recommend adding "under the Smith & Little per-participant-as-replication-unit framing" or softening to "license for small-N".
3. Edition/publisher of Macmillan & Creelman (2005) is correct as "2nd ed., Erlbaum" — but the 3rd edition (Hautus, Macmillan, Creelman, 2021, Routledge) now exists. Not an error, but if the skill is meant to compound, the 3rd edition is the more current reference.

## Per-citation review

### Kingdom & Prins (2016) — Psychophysics: A Practical Introduction, 2nd ed., Academic Press
- **Cited as:** "the primary backbone" / "the single best all-in-one modern source" for designing psychophysics experiments.
- **Actual:** Confirmed. 2nd edition, 2016, Academic Press (imprint of Elsevier), ISBN 978-0-12-407156-8. Scope: psychometric functions, adaptive methods, SDT measures, summation, scaling, model comparisons. Explicitly positioned as "the only book to combine, in a single volume, the principles… and the practical tools."
- **Rating:** FAITHFUL.
- **Recommendation:** None.

### Wichmann & Hill (2001) — two-part Perception & Psychophysics 63(8)
- **Cited as:** canonical reference for fitting + bootstrap CIs.
- **Actual:** Both parts verified.
  - Part I "Fitting, sampling, and goodness of fit", *Perception & Psychophysics* 63(8), 1293–1313.
  - Part II "Bootstrap-based confidence intervals and sampling", *Perception & Psychophysics* 63(8), 1314–1329.
  - The page range `1293–1329` the author uses collapses both parts' pages, which is reasonable shorthand.
  - Part I handles constrained ML fitting + goodness-of-fit; Part II handles parametric bootstrap CIs. Together exactly what the author claims.
- **Rating:** FAITHFUL.
- **Recommendation:** If precision matters, split into two entries with the two page ranges.

### Macmillan & Creelman (2005) — Detection Theory: A User's Guide, 2nd ed.
- **Cited as:** SDT depth.
- **Actual:** Verified. 2nd edition, 2005. Publisher in the cite is "Erlbaum"; that matches the original LEA (Lawrence Erlbaum Associates) imprint. Some catalogs list it as Psychology Press / Taylor & Francis because LEA was absorbed. Not an error.
- **Rating:** FAITHFUL.
- **Recommendation:** Consider pointing at Hautus/Macmillan/Creelman (2021) 3rd ed., Routledge, for currency.

### Rossion (2013) — "The composite face illusion", Visual Cognition 21(2)
- **Cited as:** face-specific paradigms / composite-face / "complete design" methodological argument.
- **Actual:** Full title: "The composite face illusion: A whole window into our understanding of holistic face perception." *Visual Cognition*, 21(2), 139–253. DOI 10.1080/13506285.2013.772929. This is a 100+ page review; it explicitly covers the composite paradigm, its variants, the complete vs. partial design debate, and the methodological arguments the author wants to license. The specific "complete design required for a defensible holistic-face claim" point is exactly what this review argues.
- **Rating:** FAITHFUL.
- **Recommendation:** None.

### Richler & Gauthier (2014) — Psychological Bulletin 140(5), 1281–1302
- **Cited as:** alongside Rossion, for holistic face processing + the complete-design argument.
- **Actual:** Verified. Richler, J. J., & Gauthier, I. (2014). A meta-analysis and review of holistic face processing. *Psychological Bulletin*, 140(5), 1281–1302. DOI 10.1037/a0037004. The paper explicitly argues that complete design > partial design, with the complete-design effect size roughly 3× the partial-design one, and that the two measures are not correlated. That is precisely the "complete design is the defensible one" argument the curriculum leans on.
- **Rating:** FAITHFUL.
- **Recommendation:** Note that Richler & Gauthier and Rossion disagree on interpretation in places; citing both together is fine, but a reader who reads them closely will find the literature is less unanimous than the blog implies. Non-blocking.

### Yin (1969) — face inversion effect
- **Cited as:** canonical origin of the face inversion effect.
- **Actual:** Yin, R. K. (1969). Looking at upside-down faces. *Journal of Experimental Psychology*, 81(1), 141–145. This is indeed the canonical original demonstrating a disproportionate recognition cost for inverted faces relative to non-face objects.
- **Rating:** FAITHFUL.
- **Recommendation:** The body cites "Yin (1969)" without a full bibliographic entry. If the footer list is the canonical bib, add journal/volume/pages there.

### Smith & Little (2018) — Psychonomic Bulletin & Review 25(6), 2083–2101
- **Cited as:** "the license for N=3–5 designs."
- **Actual:** Verified. Smith, P. L., & Little, D. R. (2018). Small is beautiful: In defense of the small-N design. *Psychonomic Bulletin & Review*, 25(6), 2083–2101. DOI 10.3758/s13423-018-1451-8. Volume/issue/pages all correct.
  - The paper's actual argument: small-N designs treat the individual participant as the unit of replication; power is concentrated at the within-subject level across many trials; this is valid when measurement is strong, theory is constrained, and error variance is well-controlled — as in classical psychophysics.
  - The paper does **not** defend "N=3–5" as a numeric range. It defends small-N as a class, and the cited examples from vision science commonly use 2–5 participants, but the paper's argument is about per-participant replication logic, not a sample-size cutoff.
- **Rating:** GENEROUS. The "N=3–5" phrasing compresses "small-N in the Smith-Little sense" into a specific range the paper itself does not prescribe.
- **Recommendation:** Rephrase to "license for small-N designs (with the individual participant as replication unit)" or "license for the N=3–5 sample sizes standard in vision-psychophysics, under Smith & Little's per-participant-replication framing."

### Simmons, Nelson & Simonsohn — "21-word solution" (2012)
- **Cited as:** minimum reporting-disclosure statement; year 2012; SSRN.
- **Actual:**
  - "A 21 Word Solution" is on SSRN as abstract ID 2160588, posted October 14, 2012. Year is correct (2012 for the 21-word note itself). The earlier 2011 *Psychological Science* paper "False-Positive Psychology" (Simmons/Nelson/Simonsohn, 2011, *Psych Sci* 22, 1359–1366) is the empirical foundation; the 21-word note is the 2012 follow-up.
  - The 21-word statement per the SSRN note: "We report how we determined our sample size, all data exclusions (if any), all manipulations, and all measures in the study." Check — that's the canonical wording.
- **Rating:** FAITHFUL on year/SSRN/wording. GENEROUS if the blog is implicitly leaning on the 2011 evidence base without citing it.
- **Recommendation:** Add a co-citation of Simmons/Nelson/Simonsohn (2011) to ground why the 21 words matter — otherwise a reader who looks up only the SSRN note sees a one-paragraph prescription without its motivating evidence.

## Ancillary checks

### "No Claude skill exists for perception / psychophysics / 2-AFC" claim

The author writes: "Searched GitHub for claude-skills repositories (the official Anthropic skills registry, the K-Dense-AI scientific-skills collection of 130+ skills, six separate 'awesome-claude-skills' lists cumulatively indexing thousands). Zero hits on perception, psychophysics, 2-AFC, d′, staircase, or any adjacent terminology."

Spot-checks I ran:
- K-Dense-AI/scientific-agent-skills (formerly claude-scientific-skills) exists, ~133 skills, domains include cancer genomics, molecular dynamics, RNA velocity, geospatial, time series, EEG/HRV signal processing. I did not see any psychophysics/perception/2-AFC/d′/staircase skill in the indexed content surfaced via search. Consistent with the author's claim.
- Several "awesome-claude-skills" repos exist (ComposioHQ, BehiSecc, travisvn, VoltAgent's awesome-agent-skills at 1000+ skills). The search query for those terms returned no perception/psychophysics hits.
- The claim is defensible as "appears empty on spot-check" but I did not exhaustively grep every README/SKILL.md; a deeper audit would need to clone each repo and search. For a blog-post-scale claim the spot-check is fine.

Rating: **Plausible / consistent with spot-check.** Would upgrade to confirmed only with an exhaustive search.

### Tool links (AEPsych, PsychoPy, jsPsych-psychophysics)

- **AEPsych:** `github.com/facebookresearch/aepsych` exists. Description: "adaptive experimentation in psychophysics and perception research, built on gpytorch and botorch." Python 3.10+. Active (v0.8.0 released April 2025). Matches the author's characterization.
- **PsychoPy:** `psychopy.org` is the official project site, latest 2026.1.x. Stimulus presentation and experimental control for psychology/psychophysics/cognitive neuroscience. Matches.
- **jsPsych-psychophysics:** `github.com/kurokida/jspsych-psychophysics` exists. Developer: Daiichiro Kuroki. jsPsych plugin for web-based psychophysics (gabor patches, images, SOAs, PixiJS support from 3.2.0). MIT license. Published in *Behavior Research Methods*. Matches.

All three verify cleanly.

## Final verdict

The citation backbone of the blog post is solid. Every numbered source exists at the stated venue/year/edition, and the paraphrase survives for six of eight citations at FAITHFUL. The two GENEROUS ratings (Smith & Little on N=3–5, and the 21-word note standing alone without its 2011 empirical partner) are tightening opportunities rather than misattributions — the claims are defensible but phrased slightly stronger than the primary sources warrant. Tool links and the empty-skill-landscape claim also check out. No citation would need to be pulled before publishing; one sentence softened around Smith & Little and one co-cite added for Simmons et al. 2011 would bring the whole post to FAITHFUL across the board.
