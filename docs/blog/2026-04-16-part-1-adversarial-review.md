---
status: archived
topic: archived-threads
summary: Adversarial review of "Faces as Data Mining" blog Part 1; flags synthesist frame as doing load-bearing work without empirical grounding, circular co-discovery claim, weak FFA engagement citation.
---

# Adversarial Review — Part 1

## Summary

- **15 flagged items reviewed**: 2 SOLID, 6 SOFT, 4 OVERSTATED, 2 UNDERSTATED, 1 CIRCULAR.
- **Top 3 issues the author should fix before publishing:**
  1. **The synthesist frame (§3) is doing load-bearing work it cannot support.** It's a literary metaphor escalated into a design target, then used to reject the specialist-user critique without empirical grounding. The post even says the quiet part out loud: "This isn't a compromise forced by 'we can't get experts to play a daily puzzle.' It's the actual design target." That sentence is where motivated reasoning is visible. The post needs to either demote the synthesist to analogy or commit to the empirical test *before* publishing claims that depend on it.
  2. **Claim 9.1 ("co-discovery is the *only* defensible reason to use faces at all") is near-circular with Claim 7.1.** Both rest on the same 5-task table, which is itself (per 7.3) "mostly informed guesses." The author's framework *defines* the win condition as task-4/5 performance, then observes that faces are uniquely good at task-4/5. A hostile reviewer will read this as thesis-shaped evidence.
  3. **The "different circuits" claim (6.1) is the theoretical linchpin and the citation supporting it (Tong 2012 / actually Tong & Nakayama 1999) is acknowledged as weak in the footnote.** If FFA engagement of diffusion faces is the mechanism, a weak primary source undermines the rest. The footnote confession is honest but doesn't resolve the problem: the post publishes the claim anyway.
- **Overall recommendation: fix and republish.** The post is intellectually honest and self-auditing in a way most blog posts aren't, but the honesty is concentrated in the call-out boxes. The prose between the boxes smuggles in the stronger versions of the claims. A motivated reader — including the author in two weeks — will read the prose and retain the strong form, not the hedged callouts. The hedges need to propagate upward into the headline arguments, or the arguments need to weaken to match the hedges.

---

## Pass 1 — Claim-by-claim

### Claim 2.1 — "Job-posting fraud detection, framed as EDA, reduces to 'let the user see enough of the corpus in a short enough time that unexpected structure has a chance to present itself.'"

**Rating: SOFT**

Critique: The reduction is plausible but does real work the author isn't acknowledging. It collapses "fraud detection" (which has a ground-truth label) into "EDA" (which doesn't). Fraud detection is in fact a supervised problem the moment you have one confirmed scam — and in practice scam hunters *do* have labeled examples. The post sidesteps this by framing the user as never having confirmed labels, which is a contestable framing of the actual scam-hunter workflow documented in the project's own `docs/design/scenarios.md`. The stronger, more defensible version is: "the *pattern-discovery* phase of fraud hunting is EDA-shaped." That's true and narrower. The weaker, more interesting version is what the post claims and it overreaches.

Recommendation: Either rephrase to "the pattern-discovery phase reduces to EDA" or acknowledge that once labels exist, supervised classification dominates and the tool is a pre-labeling aid, not a replacement for classifiers.

### Claim 3.1 — "The best user of a face-EDA tool is a domain non-specialist."

**Rating: OVERSTATED**

Critique: The author flags this as "plausible but untested" but the prose around it is far more confident. The post says the expert "will confirm what they expect; the player will see what's there." That's a strong empirical claim about expert cognition and it is at odds with a substantial literature on expert pattern recognition (chess masters, radiologists, wine tasters) showing that *experts see more, not less*, in their domain. Klein's naturalistic-decision-making work on firefighters and Ericsson's deliberate-practice work on experts would both push back hard. The "naive user has no priors that filter what they can notice" claim also ignores that naive users have *other* priors (what scams look like in movies, generic "shady" stereotypes) that are *worse* than domain priors, not absent.

Recommendation: Rephrase to: "A non-specialist user may see patterns an expert's priors filter *out*, but will also miss patterns an expert's priors would catch. The tool is for a different mode of pattern discovery, not a better one." That's defensible. The current framing — "the player will see what's there" — implies expertise is a net negative, which is not supported.

### Assumption 3.2 — "A synthesist-style user can build domain-relevant pattern recognition without learning the underlying domain vocabulary."

**Rating: SOFT**

Critique: The Wordle and GeoGuessr analogies don't do the work the author thinks they do. Wordle players explicitly learn English letter-frequency patterns — they absolutely learn "domain vocabulary" in the most literal sense (which letters, which positions). GeoGuessr players do learn vocabulary: they learn to name bollard types, road-line colors, utility-pole patterns. They just learn it informally from play. So the claim "without vocabulary" is already wrong for the cited analogues. What players actually do is *build procedural pattern recognition that can be articulated post-hoc*. That's a weaker, more honest claim.

Recommendation: Reframe as "players build procedural pattern recognition that is initially wordless and becomes articulable with experience" — which is empirically better-supported and still motivates the design.

### Citation 3.3 — "Watts' synthesist as a design target."

**Rating: UNDERSTATED** (i.e., the author is being more careful than necessary about the wrong thing)

Critique: The author is appropriately careful about literary citation not carrying empirical weight. What they're *not* careful about is that the synthesist is a **fictional** character with a fictional neurological condition that doesn't map to any real user population. Real non-specialists are not Siri Keeton. They have semantic processing, they form interpretations, they bring cultural priors. Borrowing the framing is fine; using it to justify design decisions ("rules out most of the obvious alternatives") is not. The footnote check says "confirm the synthesist characterization matches the novel (it does)" — but matching the novel isn't the question. The question is whether any real user resembles a synthesist, and no real user does.

Recommendation: Add a note that the synthesist is an idealization, and that real non-specialists differ from it in ways that matter — specifically, that real users bring their own filtering priors, just different ones than experts.

### Claim 4.1 — "Approaches (a)–(d) collectively cover most practical multivariate data exploration."

**Rating: OVERSTATED**

Critique: The "four families" taxonomy is too clean. Missing at minimum: (e) small multiples / trellis displays (Tufte), which are specifically designed for the "notice structure across items" task the post claims only faces serve; (f) interactive brushing / linked views (the Stasko/Heer tradition), which directly enable axis co-discovery through coordinated slicing; (g) scatterplot matrices (SPLOM), a workhorse for multivariate exploration. None of these are mentioned. The omission of brushing-and-linking is particularly awkward because the §9 co-discovery scenario ("you click a filter, the grid rearranges") is *literally* brushing-and-linking with face glyphs instead of dots — which means the novel contribution is the face substrate, not the co-discovery workflow. That's a narrower and more honest claim.

Recommendation: Acknowledge small multiples, SPLOM, and brushing-and-linking. Reframe the contribution as "face glyphs inside a brushing-and-linking UI" rather than "a fundamentally new paradigm."

### Citation 5.1 — "Scott 1992 calibration advantage."

**Rating: SOFT / needs verification** (author already flagged this)

Critique: The author has correctly flagged this as needing close reading. I'll add: the claim is doing a lot of work — it's the one piece of empirical literature that's *not* negative, and it's carrying the whole "Chernoff faces aren't actually dead" argument. If the Scott claim is paraphrased loosely or stronger than Scott actually said, the §5 argument collapses and we're back to 50 years of negative results. The footnote check says the paraphrase "may overstate" Scott — that's an admission that should propagate up. Currently §5 reads "Scott reports a specific finding" as if it's solid; the footnote says it might not be.

Recommendation: Before publishing, actually verify the Scott passage. If it turns out to be weaker than described, rewrite §5 substantially — the "cracks in the dismissal" argument is load-bearing.

### Assumption 5.2 — "Scott's finding generalizes from schematic to photorealistic faces."

**Rating: OVERSTATED as "partially supported"; it is actually UNVERIFIED**

Critique: The author labels this "unverified" which is correct, but note how §6 then proceeds as if the generalization holds and uses the FFA/N170 literature to bolster it. The structure is: "Scott says schematic faces calibrate; FFA literature says photorealistic faces engage different circuits; therefore photorealistic faces calibrate *even better*." But that's a non-sequitur — different circuit engagement doesn't imply better calibration; it could imply different failure modes, faster satiation, stronger uncanny-valley interference, etc. The argument is one-directional because it needs to be.

Recommendation: Explicitly note that "different circuit" cuts both ways — it could help calibration or hurt it; Scott's finding predicts neither outcome for photorealistic faces.

### Claim 6.1 — "Schematic vs. photorealistic faces engage different circuits."

**Rating: OVERSTATED**

Critique: The dichotomy is cleaner in the post than in the literature. FFA responds to cartoons, caricatures, and schematic faces too — Tong's own later work and the broader face-perception literature (Kanwisher's subsequent papers, Tsao's macaque work) show graded FFA response to face-like stimuli along a continuum, not a binary "icon circuit vs. face circuit." The post's framing implies Chernoff faces go to icon cortex and Flux faces go to FFA; reality is both probably engage FFA to different degrees, and the interesting question is *how much* of the face-processing stack each engages, not *whether*. The footnote acknowledges the Tong citation is weak; the claim in the body doesn't propagate that hedge.

Recommendation: Reframe as a graded claim: "Photorealistic faces likely engage *more of* the face-processing stack than schematic faces, including downstream holistic-integration components (FFA, N170, composite-face effect). The question of whether schematic Chernoff faces engage any of this stack is less settled than the dichotomy implies."

### Assumption 6.2 — "Flux can produce mismatch-territory faces along multiple axes."

**Rating: SOFT, closer to SOLID than the post admits**

Critique: This is one of the few places where the author is being *more* cautious than the evidence warrants. The r=+0.914 single-axis result is strong and the FluxSpace direction-editing literature gives plausible pathways to multi-axis control. The honest concern — whether axes stay orthogonal — is real but well-defined and measurable. This is the one assumption where I'd say the author is appropriately hedged.

Recommendation: No change needed; keep as "partially verified, Exp B pending."

### Claim 7.1 — "Faces only win at tasks 4 and 5."

**Rating: SOFT / CIRCULAR-adjacent**

Critique: The table is the author's own construction (per 7.3, "mostly informed guesses") and the claim is derived from the table. So the structure is: author writes table with their priors; table shows faces win at 4 and 5; claim concludes faces only win at 4 and 5. This is not quite circular — the table could have come out differently if the ratings had been different — but it's close enough that a hostile reviewer will call it out. The ratings in columns 1–3 for "Faces" are particularly contestable (e.g., "scan 20 cards for top N worst" at ➕ is actually testable, and the author hasn't tested it). The Borgo/Fuchs literature would predict faces should also be worse on task 2, but the post rates it ➕.

Recommendation: Either produce empirical ratings for at least one column of the table (faces on task 3, for example, is a single experiment) or weaken Claim 7.1 to "we *hypothesize* faces are uniquely strongest on tasks 4 and 5, based on the structural argument in §6."

### Assumption 7.2 — "The 5 tasks are exhaustive for EDA."

**Rating: OVERSTATED**

Critique: The author calls it "probably fine" but misses clear candidates. Missing tasks that are EDA-shaped: (6) *outlier detection* — find the item that doesn't fit the pattern, distinct from tasks 1 and 2; (7) *relationship discovery between attributes* — does attribute X co-vary with attribute Y, where neither is the label; (8) *anomaly characterization* — not just finding outliers but describing *how* they differ; (9) *corpus comparison* — this dataset vs. that dataset. Each of these is genuinely different from tasks 1–5, and at least (6) and (8) are plausible cells for faces.

Recommendation: Expand the table or narrow the claim. "5 tasks are exhaustive" is a taxonomy claim and taxonomies this clean are almost always wrong.

### Assumption 7.3 — "The ratings in the table are correct."

**Rating: UNDERSTATED**

Critique: The author says "mostly informed guesses" but the entire §7 argument depends on these ratings being *roughly* right. If even two cells flip (e.g., t-SNE+hover beats faces on task 4 because hovering shows semantic labels the face can't), the conclusion changes. The author is selling a structural argument built on unvalidated ratings as if the structure is robust to rating error. It isn't.

Recommendation: Sensitivity analysis — "which cells, if flipped, would overturn the conclusion?" — and flag those as the ones to validate first.

### Claim 8.1 — "If we reproduce the 7 Wordle/GeoGuessr ingredients, we get distribution."

**Rating: OVERSTATED**

Critique: Survivorship bias. Wordle and GeoGuessr are two successes from a population of thousands of daily puzzle attempts that didn't take off. The 7 ingredients are reverse-engineered from successes, not forward-validated on a random sample. "If we have these 7 properties we will succeed" has the same structure as "all successful startups had a pivot, so our pivot will make us successful." The 7 ingredients are plausibly necessary; the post claims they're close to sufficient.

Recommendation: Reframe as "these are plausible necessary conditions; virality remains uncertain and largely stochastic." The "we get distribution" language is promissory in a way that undermines the post's general honesty.

### Assumption 8.2 — "Text doesn't compose preattentively for 30-item scanning."

**Rating: SOLID**

Critique: This is defensible. The reading literature on chunking and parallel text processing supports the claim. The only real counterexample is expert skimming (Masson, Just & Carpenter) but the author isn't claiming it's impossible, only that it's not preattentive, which is correct.

Recommendation: No change needed.

### Assumption 8.3 — "Faces in a grid compose preattentively for 30-item scanning."

**Rating: SOFT**

Critique: The portrait-exhibition analogy is weak. Viewers of a portrait gallery have minutes to hours, not seconds, and the "gestalt impression" reports are anecdotal rather than measured. The preattentive-vision literature (Treisman, Wolfe) is fairly specific about what features compose preattentively — color, orientation, motion, size — and "subtle facial expression differences" is not on the classical list. The claim may be true, but it requires more than an art-history hand-wave.

Recommendation: Cite actual preattentive-vision work on face-specific parallel processing (there is some — Rousselet et al., and the face-pop-out literature), or acknowledge the claim is speculative on top of the preattentive-vision literature.

### Claim 9.1 — "Co-discovery is the *only* defensible reason to use faces at all."

**Rating: CIRCULAR**

Critique: This is essentially a restatement of Claim 7.1 dressed up as a new claim. 7.1 says faces only win at tasks 4 and 5; 9.1 says task 4 (co-discovery) is the only defensible use. The circularity: the structural argument that faces are good at co-discovery is (a) the face has multiple axes that (b) map to data dimensions that (c) let the user notice co-variation. But t-SNE also has multiple axes that map to data dimensions; the post says t-SNE's axes "don't mean anything" but that's only true because t-SNE doesn't plant specific semantic directions — a semantic projection (PCA along interpretable axes, LDA, supervised UMAP) *does* produce interpretable axes. The claim that *only* faces support axis discovery is wrong.

Recommendation: Weaken to "faces are one substrate for axis co-discovery; they may be a particularly good one because of the face-processing circuit's sensitivity, but they are not the only one." This is weaker but true and doesn't require defending the indefensible "only."

### Assumption 9.2 — "Co-variation invariance is achievable on Flux."

**Rating: SOFT**

Critique: Appropriately flagged as actively being measured. The risk is that Part 2 will claim cycle-consistency results and use them to close this open question, when cycle-consistency and co-variation invariance are related but not identical properties. Worth watching in Part 2.

Recommendation: No change to Part 1; watch for slippage in Part 2.

---

## Pass 2 — Unflagged assumptions

1. **"Every introductory data-viz textbook mentions Chernoff faces" (§5).** Asserted, not cited. Probably true for some definition of "introductory" but it's a historical claim used rhetorically to establish canon status. A reviewer would want a spot-check — is it in Few, Munzner, Ware, Cleveland? (Some yes, some no.) The line is doing a light rhetorical job and should either be sourced or softened to "widely taught."

2. **"The name and the modern practice both come from John Tukey's 1977 book" (§2).** Overstated. The practice predates Tukey substantially — Playfair's graphical methods, Bertin's *Semiology of Graphics* (1967), even Tufte's early work. Tukey named and codified EDA; he didn't invent the practice of looking at data. A history-of-statistics reviewer will push back.

3. **The framing of Chernoff faces as "dead" then resurrected (§5).** The rhetorical structure is "conventional wisdom says X is dead; here's why X is actually alive." This is a blog-trope that should be earned, and the evidence here (Scott 1992, unverified) is thin for resurrection. A hostile reviewer will say: you haven't resurrected Chernoff; you've proposed a different thing (photorealistic faces) and borrowed Chernoff's name for the narrative arc. That's probably fair.

4. **"The FFA activates within roughly 170 milliseconds" (§6).** FFA is an fMRI-defined region; N170 is an ERP component. The post conflates them by saying "FFA activates within 170ms, which we know from N170." fMRI does not have 170ms resolution and N170 is not a direct measure of FFA activity (its generators are debated, and extrastriate/occipital sources contribute). A cognitive neuroscientist will notice immediately. The sloppy identification is not required for the argument — the argument works fine with "face processing is fast (N170) and localized (FFA)" without welding them into a single measurement.

5. **"A scam posting can generate a face that gives the viewer a visceral 'something's off' response, where the offness is *structural* rather than aesthetic" (§6).** This distinction — structural vs. aesthetic — is introduced without definition. What makes offness structural? The perceptual-mismatch literature talks about *cue inconsistency*, not structure-vs-aesthetic. This is a terminological import from somewhere (cognitive science? design?) that isn't defended. A reviewer will ask "what do you mean by structural, and what evidence distinguishes structural from aesthetic mismatch?"

6. **"GeoGuessr has produced arguably more OSINT practitioners than every journalism school and intelligence agency combined" (§8).** Entirely unsourced rhetorical flourish. Almost certainly false if taken literally — "intelligence agencies combined" includes hundreds of thousands of analysts. The author probably means "practitioners of street-view geolocation specifically." Rhetoric like this is fine in a personal blog; in a post that polices its own claims with call-out boxes, this kind of flourish stands out as a place where the author's motivated voice slipped past the reviewer voice.

7. **"The game is itself a translation layer" (§3, five-way mapping).** The five-way mapping to the project feels retroactively fit. Items 1–5 are not derived from Watts; they're the project's design choices with Watts-shaped labels. The pattern — listing project properties, then naming them after a framework — is a place where the framework can look more explanatory than it is. A skeptical reader will ask: could you have built the same mapping onto any 5 properties of Blindsight? Probably yes.

8. **"The uncanny valley is a *signal channel*" (throughout).** This is a reframing, not a finding. The uncanny-valley literature describes it as an affective/aversive response, not as an information channel. Calling it a signal channel imports engineering framing onto a psychological phenomenon. It might be defensible — signal-detection theory can be applied — but the post treats the reframe as given when it's actually a substantive theoretical move.

9. **"The puzzle player we're designing for has no background in fraud detection, labor economics, or any particular industry" (§3).** Stated as a design fact. But the project's own scenarios document (per the CLAUDE.md) lists "scam hunter" as the priority-1 user — a domain expert. The post is redesigning around a priority-3-adjacent user (student/puzzle player) without acknowledging the conflict with the project's own stated priorities. This is a significant tension that the post papers over.

10. **"Humans are famously bad at this when the data is numbers. We're somewhat good at it when the data is pictures" (§2).** Asserted without citation and carrying a lot of weight. The "numbers vs. pictures" dichotomy is actually much more nuanced — trained statisticians do integrate numerical data, preattentive vision has known failure modes, and "pictures" vs. "numbers" is not the axis that best explains the difference. This is the kind of folk-psych claim that anchors the entire argument and that a serious reviewer will pull on.

---

## Pass 3 — Structural critique

**Structural soundness.** The argument has the following chain: §2 defines the task as EDA; §3 defines the user as a synthesist; §4 enumerates existing tools and their gaps; §5 and §6 argue photorealistic faces differ from historical Chernoff faces; §7 locates the specific task-cells where faces win; §8 extends this to consumer distribution via puzzle-game mechanics; §9 names the load-bearing mechanism (co-discovery); §10 inventories open questions. The chain is coherent in outline but brittle at specific joints.

The main structural weakness: **§3 is deployed as a filter that rules out objections before they're raised.** When §4 considers existing tools, it evaluates them against a synthesist-user model that §3 just posited. When §7 builds the 5-task table, the ratings are implicitly for a synthesist-user who prefers tasks 4 and 5. When §8 argues for a consumer puzzle, the puzzle player is a synthesist. The synthesist frame is doing the work of an axiom, and every downstream claim inherits its conditionality. If synthesist-user is the wrong target, §4, §7, and §8 all collapse. The post acknowledges this in the 3.1 call-out but then proceeds as if the acknowledgment is equivalent to resolution.

**Motivated reasoning.** Most visible in the sentence: "This isn't a compromise forced by 'we can't get experts to play a daily puzzle.' It's the actual design target." The defensive structure of that sentence ("this isn't X, it's Y") is the rhetorical shape of motivated reasoning — the author anticipates the critique that they're post-hoc rationalizing a product constraint as a product feature, and dismisses it by assertion. Whether it's *actually* post-hoc rationalization is not determined here, but the sentence is doing the work of closing that question without opening it.

Similarly motivated: the Chernoff-faces-are-dead-but-actually-alive narrative arc in §5 that rests on a single flagged-as-weak citation (Scott 1992). The narrative structure demands a resurrection; the evidence for resurrection is thin; the resurrection is nevertheless narrated as if it succeeded.

**Selection in citations.** The dismissal literature (Borgo, Fuchs, Lee) is represented fairly. But: the post cites *only* literature that either supports photorealistic-face advantages (Kanwisher, Kätsyri, Diel) or is the negative-result literature being argued against. Absent: any of the visualization-research tradition on multiple coordinated views, brushing-and-linking (Becker/Cleveland, Heer), small multiples (Tufte), or the counter-tradition in glyph design (van Wijk's table lens work, Keim's pixel-based techniques). The omissions matter because several of these traditions offer non-face solutions to exactly the tasks §7 claims only faces can do. Citing only face-relevant literature and face-dismissive literature creates an artificial binary that the rest of visualization research would resolve differently.

Also absent: any cognitive-psychology work on expertise (Chi, Ericsson, Klein) that would complicate the "expert filters out patterns" claim in §3.

**Where a hostile reviewer pushes back.**
- An ML researcher would attack Claim 6.1 first — "different circuits" is not what the face-perception literature actually says — and would also ask for evidence that Flux outputs are ecologically valid face stimuli for the N170/FFA response.
- A cognitive scientist would attack the synthesist frame as an armchair cognitive theory imported from fiction, and the "humans are bad at numbers, good at pictures" folk claim in §2.
- A visualization researcher would attack the four-families taxonomy in §4, point out that brushing-and-linking with any glyph type covers §9's co-discovery scenario, and argue that Claim 9.1's "only faces" is unsupported.
- A statistician would attack the Scott 1992 paraphrase and the "calibration advantage" framing as possibly overselling what Scott wrote.
- A game designer would attack Claim 8.1 as survivorship-biased and note that the daily-cadence + shareable-grid pattern has been tried for many topics since 2022 and most failed.

**Synthesist-as-shield.** This is exactly the issue raised in the prompt. The synthesist frame is used in §3 to pre-emptively rule out the most obvious critique of face-EDA — that experts perform worse with it than with their existing tools. The framing says: if experts do worse, that's because they're experts; we wanted non-experts anyway. This is unfalsifiable in structure: no empirical result about expert performance can threaten the thesis, because the thesis has redefined the user. That's a serious issue, and the 3.1 call-out ("plausible but untested") does not resolve it — it only flags it. If the synthesist frame is load-bearing, the empirical test (naive vs. expert on detective runs) needs to happen *before* publication of claims that rest on it.

---

## Optional — citation spot-check

Flagging for web verification in order of importance:

1. **Scott 1992 "calibration advantage."** Author already flagged. The specific concern: the "face-glyph encodings produce calibration-advantaged viewing in users who look at many faces in sequence" formulation is remarkably specific and doesn't match the style of Scott's actual text (which is a density-estimation textbook). Worth checking whether Scott said anything this specific or whether it's a generous paraphrase of more general observations. If the latter, §5 needs substantial rewriting.

2. **Tong & Nakayama 1999 / "synthetic faces activate FFA."** The footnote (labeled `tong2012` but actually citing a 1999 paper) is already flagged as weak. The 1999 paper is about visual search with face stimuli, not about FFA activation of synthetic faces specifically. The footnote label mismatch ("tong2012" → 1999 paper) also suggests the author swapped citations without a clean pass. Worth either finding a better primary source or softening Claim 6.1.

3. **Young 1987 composite-face effect.** Author lists as uncontroversial. Correct — this is the canonical original demonstration. However, the modern interpretation of the composite-face effect as evidence for "holistic processing" is contested; Richler, Cheung, and Gauthier have argued the effect is better explained by spatial attention and decisional factors. The post takes the holistic interpretation as settled. Worth a sentence acknowledging the controversy.

4. **Kanwisher 1997 / Bentin 1996.** 170ms figure for N170 is correct. FFA localization is correct. The conflation of the two (fMRI region and ERP component reported as one measurement) is the problem, not the individual citations. See unflagged assumption #4 above.

5. **Kätsyri 2015 and Diel 2022 "perceptual mismatch."** The "mismatch not uniform degradation" framing is a fair summary of Kätsyri's review conclusion. Diel's meta-analysis supports mismatch but is more equivocal than the post implies — the meta-analysis found mixed evidence for several mismatch sub-hypotheses and the "mismatch is the mechanism" reading is the author's synthesis, not the paper's conclusion verbatim. Worth checking whether the paraphrase "confirming mismatch mechanism; uniform degradation does not trigger the valley" is faithful to Diel's conclusions section.

---

## Final verdict

The post makes a coherent case for an interesting research direction, and its self-auditing structure is genuinely better than most blog writing in this genre. But the case is weaker than the post presents. Three problems recur. First, the hedges live in the call-out boxes while the prose runs on the un-hedged versions of the claims — a reader will retain "photorealistic faces engage different circuits" and "co-discovery is the only reason to use faces," not the "plausible but uncertain" qualifiers. Second, the synthesist frame in §3 is doing more work than a literary analogy can support, and it functions as an unfalsifiability shield against the most obvious critique. Third, several load-bearing citations (Scott 1992, Tong 1999) are flagged-as-weak in their own footnotes but the claims built on them are not correspondingly weakened in the body. The post is not wrong, but it is overconfident in exactly the places it thinks it is being careful. Fix and republish: tighten the prose claims to match the call-out hedges; either validate or demote the synthesist frame; verify the Scott citation before leaning on it; acknowledge the visualization traditions (brushing-and-linking, small multiples) that already solve parts of what the post claims only faces solve.
