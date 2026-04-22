---
status: archived
topic: archived-threads
summary: Adversarial review of "Faces as Data Mining" blog Part 2; criticises task-anchored cycle as rhetorical, loss degradation story too rosy, DRaFT-1 commitment underjustified for external projection.
---

# Adversarial Review — Part 2

## Summary

- **11 flagged items reviewed**: 1 SOLID, 4 SOFT, 4 OVERSTATED, 1 UNDERSTATED, 1 CIRCULAR.
- **Top 3 issues the author should fix before publishing:**
  1. **§3's "task-anchored cycle" reframe is largely rhetorical.** Once you strip the terminology, the construction is a standard supervised end-to-end pipeline (text → ... → predict D) with a loss between input label and output prediction. That is not "cycle consistency"; it is cross-entropy on a labeled supervised problem. Calling `CE(D, D̂)` a "cycle" imports the vocabulary and citation-prestige of CycleGAN while the actual mathematical content is ordinary supervised learning. The "cycle" framing is borrowed goodwill from a neighboring literature. The prose should either commit to the actual construction (a supervised task with an information-preservation regularizer) or justify the cycle import beyond vocabulary.
  2. **Claim 6.2 ("loss degrades gracefully from labeled to unlabeled") glosses the hardest case.** The labeled→unlabeled story assumes that InfoNCE + VICReg on `(qwen, Ψ(Flux(P(qwen))))` pairs will preserve *the structure that would have been the task*. But in the unlabeled regime there is no task: InfoNCE + VICReg will preserve *whatever dominates the variance* of qwen output, which for scam corpora is likely language/topic/length structure, not suspicion structure. The author knows this (failure mode 5) but treats it as a remediable edge case rather than the default behavior when the task signal isn't dominant. The synthesis doc itself (`docs/research/2026-04-16-full-loop-synthesis.md` §2.4) has a stronger recipe — InfoNCE + soft-rank Spearman + VICReg, three-term — and §6.3 of the post explicitly demotes soft-rank to "kept warm, not exercised." That demotion is at odds with the synthesis.
  3. **The DRaFT-1 commitment (Claim 7.1, Stage 4) is underjustified for this specific pipeline.** DRaFT's published results are on full LoRA or UNet fine-tuning, where the gradient flows through all of Flux's parameters. Here, the only thing being updated is a small external projection `P` in the conditioning layer — a much shorter gradient path where the one-step approximation of intermediate activations matters more, not less. The author argues the opposite ("coarse attribute axes, which should be robust to the bias") without cited evidence. This could easily be the load-bearing wrong decision of the whole staged plan.
- **Overall recommendation: fix and republish.** The post is technically serious and well-annotated, but the §3 reframe is doing rhetorical work that the math doesn't support, the §6 loss story is cleaner than the problem actually is, and the §7 gradient-method commitment needs either a smaller ante (ship Stage 3 hand-written regression first, as the synthesis actually recommends) or a better citation story.

---

## Pass 1 — Claim-by-claim

### Claim 1.1 — "A machine-evaluable target correlating with human co-discovery is feasible."

**Rating: SOFT**

Critique: The claim is hedged ("empirical question we plan to measure") but the hedging papers over a real problem: the target the post constructs (`D̂ vs. D`) is mechanically measurable only for *planted* patterns in the detective corpus. In the scam/real-data regime — which Part 1 claimed was the actual user scenario (priority-1 user is the scam hunter per `CLAUDE.md`) — there is no `D` and the target is structurally unavailable. The post defers this to §6 Claim 6.2 and §8 Stage 6, but neither resolves it. The target is feasible only in exactly the synthetic regime; the whole framing inherits Part 1's pivot from the priority-1 user to the synthesist/puzzle-player.

Recommendation: Clarify that the machine-evaluable target depends on label availability; in the real-data setting, the unlabeled-regime loss is a *different target* and its correlation with human co-discovery is a separate (and harder) empirical question.

### Assumption 2.1 — "Conditions (1) and (2) are nearly decoupled from (3)."

**Rating: OVERSTATED**

Critique: "Likely but not guaranteed" is the footnote-hedge, but the claim is doing real work: it's the license for optimizing against machine metrics during development. The decoupling assumption fails in well-known ways for face encoders — adversarial perturbations, watermarking artifacts, and (relevant here) the FaRL/CLIP encoders explicitly learn human-language-aligned features rather than human-perceptual ones. A CLIP-derived encoder tuned on alt-text from the web will score "a smiling businessman" high on any face the captioner would describe that way, regardless of whether a human at a glance would read it as smiling. That is a gap between machine-distinguishable and human-distinguishable, not an alignment. The "trained on human-perceivable face attributes" framing assumes the training signal was human perception, but FaRL's training signal is image-caption contrastive learning — weaker.

Recommendation: Downgrade to "hoped rather than assumed" and name one or two specific ways the decoupling is known to break. The Chu et al. steganography result cited elsewhere in the post is exactly this kind of failure.

### Claim 3.2 — "Task-anchored cycle is strictly more informative than embedding-level cycle."

**Rating: CIRCULAR (and relabel-only)**

Critique: The author's own hedge ("mild claim... almost a tautology") understates the problem. The "task-anchored cycle" is not a cycle. A cycle is a round-trip that returns to the same space: `A → B → A`. Here, `D → text → qwen → P → Flux → face → Ψ → R → D̂` does return to `D`-space, but only because the author has inserted a learned readout `R` whose entire job is to land back in `D`-space. That's not cycle consistency; it's an encoder-decoder with a cross-entropy loss, which is the default architecture for any supervised classification problem. The CycleGAN import buys nothing here — CycleGAN's innovation was training without pairs, via *two* generators and *two* cycle losses; the post has one direction and labeled pairs. Calling `CE(D, D̂)` a cycle gets CycleGAN's citation-prestige while doing supervised learning.

The "more informative than embedding-level cycle" comparison is a straw man: embedding-level cycle was never the natural framing for a labeled cross-modal classification problem. The author sets up a weak alternative, then beats it.

Recommendation: Drop the "cycle" terminology for the task-anchored case or justify the import. The honest description is "a supervised cross-modal classification pipeline with an information-preservation structural loss." The actual cycle content — round-trip *without* labels — belongs in the §6.2 unlabeled regime and isn't explored.

### Assumption 4.1 — "FaRL beats ArcFace for our axes."

**Rating: SOFT**

Critique: Reasonably defended by the CelebA probe numbers, but two things get glossed. First, the published FaRL lift over vanilla CLIP is +0.5 mAcc — small. The synthesis memo is honest about this ("the lift is small but structural") but the post rounds up to "a better choice." Second, the synthesis explicitly keeps ArcFace for the signature-cluster axis and recommends *ensembling* FaRL + DINOv2; the post's §4 treats FaRL as the primary and DINOv2 as "optional," which is a weakening of the synthesis recommendation without argument. Also, FaRL's training domain is LAION-Face — caption-weighted — which is not the domain of the detective-corpus faces. Distribution shift at inference is a real concern the post doesn't mention.

Recommendation: Commit to the ensemble recommendation the synthesis actually made, not a softer version. State the CelebA benchmark's domain shift from the detective corpus.

### Claim 5.1 — "The six metrics form a non-redundant diagnostic panel."

**Rating: OVERSTATED**

Critique: The claim is that each metric measures "a property the next one doesn't." That's argued structurally, not measured. Several of the metrics are known to correlate heavily in practice:
- k-NN overlap and trustworthiness/continuity are computed from the same rank structure. Kraemer et al. (2018) showed trustworthiness is essentially a smoothed k-NN Jaccard at larger k. They are not independent diagnostics; they are two parameterizations of the same quantity.
- Linear CKA and k-NN overlap are both measuring local geometric alignment; Kornblith et al. show CKA correlates with retrieval accuracy (a k-NN relative).
- ROC-AUC on cluster pairs and `D̂ vs. D` classification accuracy are trivially related when `D` is near-binary.

A truly non-redundant panel would need an explicit redundancy analysis (how correlated are these metrics across a sweep of deliberately-perturbed pipelines?). The post treats structural argument as sufficient. A panel of 6 metrics measuring 3 underlying properties is fine but the diagnostic power is overstated.

The GW metric is also slipped past: published GW on batches of hundreds is expensive and entropic GW's regularization choice (ε) is a free parameter that materially changes the score. That's a research project, not a diagnostic.

Recommendation: Either run a redundancy analysis on v3 data (the data exists) or rank the six metrics into a tiered panel, acknowledging which pairs are likely redundant.

### Assumption 5.2 — "Metrics transfer to cross-modal setting without losing diagnostic interpretation."

**Rating: SOFT**

Critique: The partial-support framing is honest, but the specific concern is larger than admitted. The source memo (per §5 caveat) is about "two text encoders on the same face domain." The post moves to "one text encoder and one face encoder on the same text domain." Those are different problems in a specific way: in the source setting, both maps are onto the same manifold (faces), so the neighborhood structure is meaningful because neighbors in one encoder should be spatially close on the target manifold in both. In our setting, the two spaces have different topologies (text-embedding space is not diffeomorphic to face-encoder space), so "neighborhood preservation" can only mean "rank preservation under some learned correspondence," and that's what the readout is. The k-NN overlap metric is thus measuring something more like "does the learned mapping preserve neighborhoods" than "do the two spaces agree on neighborhoods" — a different quantity.

Recommendation: State which metrics measure a *property of the pipeline* (rank preservation across a learned map) vs. a *property of two aligned spaces* (neighborhood agreement between encoders), and note that the source memo's metrics are for the latter.

### Claim 6.2 — "Loss degrades gracefully from labeled to unlabeled."

**Rating: OVERSTATED**

Critique: See top-line issue #2. "Graceful degradation" assumes the InfoNCE + VICReg remainder preserves the task-relevant structure. In fact, Daunhawer's identifiability result (Citation 6.1) recovers the *shared content block* between modalities — whatever is shared, not whatever is useful. On a scam corpus, the largest shared axes between qwen and FaRL are almost certainly *domain* (what industry, what format, what language) — not suspicion. The unlabeled loss will faithfully preserve domain structure and be silent on suspicion, which is the opposite of what the project wants.

The post acknowledges this as failure mode 5 ("InfoNCE finds degenerate shared content") but the remedies are weak: "regularize more aggressively" is not a specification, and "use heuristic labels" admits the unlabeled regime doesn't actually work and pulls in pseudo-supervision. That's fine, but it's not "graceful degradation" — it's "unlabeled regime requires different infrastructure."

Also: the synthesis doc (source of the loss function) explicitly lists three non-task terms (InfoNCE, *soft-rank Spearman*, VICReg). The post drops soft-rank to "optional" (Assumption 6.3). Why? The synthesis lists it as a "cheap baseline" first-class term, not an optional one. Downgrading the structure term is the specific place the claim breaks: soft-rank is the term that would preserve *geometry*, which is what survives when task signal is absent.

Recommendation: Either restore soft-rank as a first-class term (align with the synthesis) or argue explicitly why the post diverges. As written, Claim 6.2 understates the hardness of the unlabeled regime and omits the term that would help most.

### Citation 6.1 — Daunhawer et al. 2023, identifiability.

**Rating: SOFT / possibly OVERSTATED**

Critique: The Daunhawer ICLR 2023 paper shows identifiability of the shared content subspace *under specific assumptions*: content-style independence, appropriate encoder capacity, infinite data. The post's paraphrase — "provably and identifiably recovers the shared content block" — reads as if the recovery is automatic. It isn't. The assumptions are strong and not self-evidently satisfied in a qwen-embedding × generated-face pair. Specifically, "content-style independence" requires that the style variation in qwen (word choice, phrasing) is statistically independent of the style variation in face-space (lighting, background, hair). That independence holds by construction if you generate faces independently of qwen text — but the whole pipeline is explicitly *conditioning faces on qwen*. If Flux faces' style depends on qwen text (they do, that's the point), content-style independence is violated and Daunhawer's result doesn't apply cleanly.

The author's "check" ("plausibly a reasonable fit but not explicitly verified") flags this but doesn't resolve it. The paper's result is being loaded as "the mathematical justification" for InfoNCE — which is a stronger load than "this is a useful heuristic, bolstered by related identifiability work."

Recommendation: Soften to "motivated by identifiability work in related settings." Acknowledge that content-style independence is non-obvious in the conditioned-generation setting and likely violated.

### Claim 7.1 — "DRaFT-1 sufficient for detective-scale pipeline in 3–5 days."

**Rating: OVERSTATED**

Critique: See top-line issue #3. DRaFT's published results are on SDXL/SD reward fine-tuning, where gradients propagate through the full diffusion stack and the one-step backprop is a truncation of a much longer signal. Here, the only learnable parameter downstream of Flux is absent (Flux is frozen) — the learnable parameters are `P` (upstream of Flux) and `R` (downstream of Ψ). The DRaFT-1 gradient reaches `P` only through one denoising step of Flux, which is a *much shorter* gradient path than the published setups. Whether that one-step signal is sufficient to train an *input projection* is a different question from whether it's sufficient to fine-tune output denoising.

The failure mode the author names ("P learns to produce images that trick the readout") is real but is not the main concern. The main concern is that one-step-DRaFT through a frozen 12B-param Flux produces a nearly-random gradient direction for a projection layer far upstream of the evaluation point; the gradient's coherence is not a given. The synthesis memo *itself* lists TexForce as the structurally similar setup (Citation 7.3), which uses DDPO not DRaFT. That's a discrepancy the post glides over: the closest structural precedent doesn't use DRaFT at all. The post flags this with "DDPO won't transfer because ODE-vs-SDE" but that only tells us why TexForce's method is incompatible — it doesn't tell us DRaFT is sufficient as the replacement.

Also: "3–5 days on a single 24GB GPU" is not cited. The synthesis memo says ~3–5 days *including implementation*, which is quite different from training time — implementation alone for DRaFT-1 on Flux-dev's specific sampler path is probably 3 days, leaving 0–2 days for training. Possibly fine; not obviously fine.

Recommendation: Weaken to "DRaFT-1 is the cheapest reasonable start, but there is no published precedent for applying it to a frozen-Flux-plus-upstream-projection architecture." Also: recommend Stage 3 (hand-written conditioning + supervised regression on P) as the actual first-line approach, consistent with the synthesis, which explicitly flags it as "the biggest pre-RL win" and bypasses Flux gradients entirely.

### Assumption 7.2 — "DDPO/DPOK diverge on rectified-flow samplers."

**Rating: SOLID**

Critique: Correctly attributed to Flow-GRPO and DanceGRPO papers. The ODE-vs-SDE issue is technically real and well-characterized. The gotcha-flagging is valuable.

Recommendation: No change.

### Assumption 6.3 — "Soft-rank structure term is optional."

**Rating: UNDERSTATED in the wrong direction** (the author is being more confident than warranted that it can be deferred)

Critique: The synthesis doc's loss recipe is four-term (CE when labels exist + InfoNCE + *soft-rank Spearman* + VICReg). The post's is three-term (CE + InfoNCE + VICReg) with soft-rank demoted to "kept warm." The demotion is presented as "starting simpler" but the simpler form loses the specific term that does structure preservation. VICReg does variance-floor and decorrelation, not rank preservation. InfoNCE does contrastive alignment, not global geometry. Dropping soft-rank leaves the loss without any explicit geometry-preserving term, which is exactly the term needed when CE is absent (the unlabeled regime). This is an unforced narrowing.

Recommendation: Restore soft-rank as a first-class term or explicitly argue why two-term is sufficient. The "simpler starting point" reasoning cuts against the synthesis and the author doesn't explain why.

### Claim 9.1 — "The eight failure modes are roughly exhaustive."

**Rating: OVERSTATED**

Critique: "Exhaustive by construction, not by experience" is self-aware but the construction is family-level, not mode-level, and several families are conspicuously absent. Missing from the inventory:
- **Reward hacking** (distinct from DRaFT-1 trick-the-readout): the generator finds a low-dimensional image subspace that maxes the probe without producing faces on-manifold. Named "adversarial face" is the standard term.
- **Catastrophic forgetting of Flux's prior** — even with Flux frozen, if `P` produces out-of-distribution conditioning, Flux generates non-face outputs. Hand-written conditioning sanity check (Stage 3) partially addresses, but not as a named failure mode.
- **Data leakage between stages** — the same 543-item corpus is used for v3 baseline, Stage 1 probe, Stage 4 training. Without disciplined splits, Stage 4 overfits to held-out-in-name-only data.
- **User-experiment confound**: the detective-experiment users may learn the researcher's biases about what patterns are "findable," not the planted patterns. This is the analog of "teaching to the test" and is specifically relevant because the detective corpus is generated by the same pipeline being evaluated.
- **Non-stationarity of the face-encoder during evaluation** — Ψ (FaRL) is frozen but its weights are from a specific checkpoint; if the encoder has dataset-specific artifacts (LAION-Face artifacts show up on in-domain vs. out-of-domain faces differently), the metric values aren't comparable across corpora.

Five additions to an "exhaustive" list of eight is a lot. The list is reasonable for a planning document but not for a claim of exhaustiveness.

Recommendation: Relabel "eight named failure modes covering the main families" and remove the exhaustiveness claim. Add reward hacking, data leakage, and user-confound at minimum.

---

## Pass 2 — Unflagged assumptions

1. **"Flux.1-dev takes a 4096-dimensional text-conditioning tensor (roughly — Flux has multiple conditioning surfaces, a detail we're glossing)" (§4, Link 2).** The parenthetical acknowledges this but the rest of §4 and all of §7 proceed as if `P: ℝ^1024 → ℝ^4096` is well-defined. The synthesis memo explicitly flags this as open question #3: Flux has T5-XXL text embeddings, CLIP-L text embedding, pooled CLIP, and a learned guidance vector — each a different `P` target with different gradient paths. The post papers over a decision that isn't made. An engineer who has trained diffusion models will notice immediately that "Flux conditioning" is a plural noun.

2. **"Frozen" is used seven times as if it were an operational primitive.** Flux is "frozen," Ψ is "frozen," infrastructure is "frozen." But Flux-dev's distilled guidance-embedding path is a known-unknown for gradient methods (synthesis open question #3). "Frozen" in the DRaFT-1 setup means "no parameter updates," not "no gradients flow." The gradients through the denoising step require differentiating through all of Flux's attention layers, which on a 24 GB GPU is exactly the memory bottleneck DRaFT-1 is supposed to solve. "Frozen" elides the difference between "no parameters trained" and "no compute or memory cost." An engineer knows the difference; a reader may not.

3. **The step from "r(sus)=+0.914 on one axis" to "we can train P to achieve multi-axis structure preservation" (inherited from Part 1's 6.2 assumption and smuggled into §3/§7 here).** Part 1's assumption 6.2 was "partially verified (r=+0.914 on 1 axis)" — but Part 2 treats multi-axis preservation as an engineering target (Stage 2 exit criterion: "all 3 orthogonal axes + interaction cell + signature cluster recover at ≥70%") without revisiting the evidence. The single-axis result becomes the multi-axis premise. Part 1's review explicitly flagged this ("the risk is that Part 2 will claim cycle-consistency results and use them to close this open question"); Part 2 hasn't closed it but it has structurally assumed it.

4. **The synthesist/puzzle-player user is implicit in "labels exist in the detective experiment; they don't exist in real-data corpora."** The whole labeled→unlabeled framing of §6 presumes the labeled regime is the detective puzzle (synthetic) and the unlabeled regime is real data. But the priority-1 user from `docs/design/scenarios.md` is a scam hunter with *real* labels (confirmed scams accumulated over time). The post has silently promoted the puzzle-player/synthesist user over the scam hunter, which is exactly the pivot Part 1's review flagged as motivated. Part 2 smuggles it in by treating "labels" as puzzle-labels only.

5. **"The obvious candidate is ArcFace IR101... Recent research... argues this is the wrong choice" (§4).** The framing implies an emerging-consensus that ArcFace is wrong for this kind of task. In fact, the "recent research" is a synthesis of arXiv:2507.11372 (the project's own research memo, not an independent finding) and the FaRL paper (which doesn't discuss ArcFace at all). The narrative structure — "obvious baseline" → "recent research overturns it" — is rhetorical, not factual. ArcFace is the wrong tool *if* your axes are expression/attire/gaze; it's the right tool *if* your axes are identity. The post doesn't acknowledge that choosing axes to be the non-ArcFace ones is a design choice that makes the "switch to FaRL" finding look inevitable.

6. **"Small networks, both differentiable" (§7).** P and R are both described as small. But "small" is a stand-in for "trainable on a 24 GB GPU through DRaFT-1's gradient path." If P is a single linear layer ℝ^1024 → ℝ^4096, that's 4M parameters, trivially small. If P is a small MLP, it's 20–50M parameters. If the conditioning surface is really four separate tensors (T5-XXL ~4096-d, CLIP-L 768-d, pooled 768-d, guidance 256-d), a P that targets all of them is meaningfully larger and the gradients-through-Flux are four separate paths. The "small network" framing defers the architecture decision without flagging it.

7. **"Five stages... each cheaper than the next and each either validating or killing the expensive stages that follow" (§8).** This is a pre-registered-Bayesian framing — each stage's result updates belief, and a failure should kill downstream stages. But Stage 1's failure mode (probe accuracy is low) doesn't cleanly kill Stage 4 — you might decide the problem is probe architecture, not the pipeline, and proceed anyway. The "kill" discipline requires pre-committing to specific thresholds. The post lists thresholds only for Stage 1 ("≥80% high, <50% low") and Stage 2 ("≥70% per-axis"); Stages 3–6 don't have exit criteria in the post (though the synthesis lists some). Without thresholds, the staging is aspirational.

8. **"The machine-evaluable target" is singular throughout the post.** In fact there are multiple targets: (a) per-metric values in §5, (b) the CE+InfoNCE+VICReg loss in §6, (c) per-axis recovery rate in §8. These are related but non-equivalent, and optimizing (b) doesn't guarantee (a) or (c). The post's elision makes the "target" feel unified; in engineering practice, you'll be navigating between these three targets and their disagreements.

9. **"Microsoft publishes reference code for Flux/SD3" (§7, Adjoint Matching).** The post lists this as if it were ready-to-use. The synthesis memo says the same thing. But what a repo "publishes" and what works on your specific fine-tuning task are different. The Microsoft repo is for reward fine-tuning of whole-model or LoRA setups; porting it to "only update an upstream projection P while keeping everything else frozen" is non-trivial and is listed as "more engineering effort to port" in the synthesis. The post's phrasing ("Microsoft publishes reference code ... **Use when DRaFT plateaus**") is more decisive than the synthesis warrants.

10. **"We're starting simpler and escalating if needed" (§6.3).** This is a reasonable framing for *engineering*, but the claimed four-term synthesis recipe → three-term blog-post recipe is a narrowing of the prior art without stated reason. The synthesis explicitly did the work of arguing four terms; the post silently drops one. "Simpler" here means "dropped the term I didn't want to implement," which is a different kind of simpler.

---

## Pass 3 — Structural critique

**Is §3's "task-anchored cycle" the organizing idea?** No. The construction in §3 reduces to: train a supervised classifier `R ∘ Ψ ∘ Flux ∘ P ∘ qwen` on (text, label) pairs, with CE loss. That's a standard end-to-end classification pipeline with a non-trainable middle (Flux). The "cycle" label adds nothing that standard supervised-learning vocabulary doesn't already cover. The rest of the post — §4 pipeline decomposition, §5 diagnostic metrics, §6 contrastive+collapse regularization, §7 gradient methods — is a perfectly reasonable cross-modal pipeline document. It doesn't need §3's framing to hold together; §3 is a rhetorical shell. Delete §3, retitle §5 "measurement" and §6 "training objective," and the post reads as "a cross-modal supervised pipeline with structural auxiliaries," which is what it is. The CycleGAN citation becomes unmotivated. This is close to structurally identical to the Part 1 Blindsight-synthesist issue: a vocabulary borrowed from a neighboring literature that doesn't carry the weight it's asked to carry.

**Section ordering.** §6 (loss) and §7 (gradients) are in the right order, but §5 (metrics) and §6 (loss) have a subtle dependency that isn't surfaced. Several §5 metrics (linear cycle residual, k-NN overlap, CKA) are *related to* the structure-preservation terms in §6 (VICReg, soft-rank). Specifically: the soft-rank Spearman term in §6 is a *differentiable relaxation* of the rank-correlation underlying trustworthiness/continuity in §5. If the post had that connection, the "metrics" and "loss" sections would cohere; they currently read as parallel lists. The synthesis memo makes the connection; the post loses it. §6.3's demotion of soft-rank reads doubly strange once you see that the demoted term is a differentiable version of §5's rank-preservation metric.

**Eight failure modes — real inventory or retrospective justification?** Partly retrospective. Modes 1–4 are directly mapped to decisions already made in the pipeline (qwen choice, Flux choice, FaRL choice, DRaFT-1 choice); each "remedy" is the next-step-up in the staged plan. This is not quite a risk inventory — it's a decision tree for the staged plan, relabeled as failure modes. Modes 5–7 are more independent (objective misspecification, generalization failure, sampler change). Mode 8 is the human-experiment acknowledgment. The omissions (see Pass 2 #7 and Pass 1 Claim 9.1) are specifically at the *interaction* level — what happens when two decisions interact badly — which is exactly what risk inventories miss when they're structured around already-made decisions.

**Where a hostile reviewer pushes back first.**

*(a) ML researcher with diffusion-model background.* Attacks §7 immediately. "DRaFT-1 on a frozen-Flux upstream-projection fine-tune is not the published setup; cite the work that does this." Probably also attacks the Flux conditioning-surface handwave — "which surface? T5 or CLIP? pooled or token? guidance?" And the 4096-d claim is off (Flux T5-XXL output is 4096, but there are also pooled CLIP of different dim and guidance embedding; saying 4096 glosses the real plural).

*(b) Representation-learning theorist (Daunhawer-style).* Attacks Citation 6.1. "The content-style independence assumption is not shown; in conditioned generation, style and content are explicitly coupled; your identifiability result doesn't apply. Cite Daunhawer's Section 4 assumption list and check each."

*(c) Visualization/HCI researcher.* Attacks Claim 1.1. "Your target correlates with human co-discovery only if humans can read the face-encoder's decisions. You haven't studied face-encoder outputs psychophysically. You're optimizing against a machine-measurable target that may diverge from human-readable differences. Add a human-in-the-loop evaluation before committing."

*(d) Engineer who has trained diffusion on 24GB.* Attacks §7 and §8. "DRaFT-1 *through Flux-dev* requires differentiating through one denoising step of a 12B-parameter model. On 24GB that's tight; on 24GB with batch size >1 it's infeasible. You haven't specified batch size or precision (bf16? fp8?). 3–5 days is implausible if each batch takes a minute; how many gradient steps is 3 days? What's the LR schedule? Have you checked that a single-step backprop produces a usable gradient in a projection layer that's in the upstream conditioning space?" Also attacks Stage 3: "why isn't this stage 1.5? Hand-written conditioning regressed against your target is the obvious first thing, as the synthesis memo says."

**Motivated reasoning.** Three visible places.

- **§3's "task-anchored cycle" terminology.** The author wanted to keep the cycle-consistency framing from Part 1's technical lineage (and from the project's framework v0.11, which uses cycle-related properties). But the actual construction is supervised classification. The cycle vocabulary is retained against the math. This is motivated: if the "cycle" framing goes, the post is less distinctive and more obviously "supervised cross-modal classification."

- **§6.3's downgrading of soft-rank.** The synthesis recommends four terms; the post uses three. The demoted term happens to be the hardest to implement (soft-rank relaxations are fiddly) and the one most responsible for unlabeled-regime generalization. "Kept warm, not exercised" is the language of decisions-already-made being justified post-hoc.

- **§7's commitment to DRaFT-1 over Stage 3 (hand-written + regression).** The synthesis memo explicitly calls Stage 3 "the biggest pre-RL win" and recommends it before DRaFT. The post buries hand-written-conditioning as "sanity check" and stages DRaFT-1 at Stage 4. The staging makes DRaFT-1 look like the main event, when the synthesis suggested the main event is regression. This is motivated in the direction of narrative — DRaFT-1 is more impressive to describe — against what the literature review recommended.

---

## Optional — citation spot-check

Flagging for verification in order of importance:

1. **Daunhawer et al. ICLR 2023, identifiability.** The "provably and identifiably recovers the shared content block" phrasing is strong. The paper's actual result (per my reading recall) is subspace-level identifiability under content-style independence, and the independence assumption is discussed at length in the paper's Section 3–4. The post's "plausibly a reasonable fit" hedge is accurate but the body prose is stronger. High-risk citation for the load it's carrying; recommend the author re-read Daunhawer's assumptions section specifically.

2. **DRaFT (Clark et al. 2023).** The post describes DRaFT as "works empirically for reward-alignment in published work on SD and related models. Start here." The paper's published results are on SD and SDXL. The synthesis memo correctly notes "yes, trivial" for Flux-readiness, but "Flux-ready" means "the technique *is transferable to* Flux," not "the published results are on Flux." The post's phrasing conflates these. Verify: does any published DRaFT result apply to a frozen-Flux-plus-upstream-projection setup? My read: no. The post should state this.

3. **Adjoint Matching (Domingo-Enrich et al. ICLR 2025).** Footnote claims Microsoft reference code at `github.com/microsoft/soc-fine-tuning-sd`. Verify this repo is: (a) publicly accessible, (b) actually contains Flux/SD3 code (not just SD/SDXL), (c) actively maintained as of 2026-04. The synthesis memo cites it without verification; the blog post inherits the unverified claim and presents it as "working code." Low-risk if true; high-risk if the repo is stale or SD-only.

4. **CycleGAN (Zhu et al. 2017) import.** See Claim 3.2 critique. The import is citation-only — CycleGAN's actual method (two generators, two discriminators, two cycle losses, unpaired data) does not match the post's "task-anchored cycle" (one pipeline, one CE loss, paired labeled data). The citation is background, not mechanism. The prose leans on it more than the mechanism supports. Not a misquote, but a misuse.

5. **Flux paper (Esser et al. arXiv:2403.03206).** The paper is the Stable Diffusion 3 / MMDiT paper. Flux.1-dev is derived from that line (Black Forest Labs, ex-SD3 team) but is not the same model — SD3 is rectified-flow, Flux-dev is rectified-flow-with-guidance-distillation. The post's "Flux.1-dev is a rectified-flow transformer" is mostly-correct but glosses the guidance distillation, which is the specific feature that makes Flux-dev's gradient behavior non-standard (and which the synthesis memo open-question-3 flags). Citing Esser et al. for "Flux is rectified flow" is close-enough but not precise; for gradient-method arguments (§7), you want a Flux-specific reference, and there isn't a great one because BFL hasn't published a Flux paper.

6. **Kraemer et al. / trustworthiness-k-NN relationship.** Not cited by the post; I cited it in Claim 5.1 from recall. Author should verify before relying on the non-redundancy-critique pushback. This is my citation, not theirs, and may be misremembered — it's a claim about metric redundancy that the author should check if they want to defend the six-metric panel.

---

## Final verdict

Part 2 is substantively better-sourced than Part 1, but it has two characteristic weaknesses: it inherits Part 1's pattern of letting vocabulary (synthesist, cycle) do more work than the underlying construction supports, and it narrows the synthesis memo's recommendations in motivated ways (soft-rank demoted, hand-written regression deferred, Flux conditioning surface left ambiguous). The "task-anchored cycle" is not a cycle in any non-trivial sense and the CycleGAN import is vocabulary borrowing; the labeled→unlabeled "graceful degradation" ignores that the unlabeled loss's dominant signal is domain structure, not task structure; and the DRaFT-1 commitment treats published SDXL results as directly transferable to a frozen-Flux upstream-projection setup without argument. Fix the §3 framing (commit to "supervised cross-modal pipeline with structure regularizer" or justify the cycle import), restore soft-rank as a first-class loss term consistent with the synthesis, and demote DRaFT-1 to "Stage 4, if Stage 3 regression is insufficient" — which is what the synthesis actually recommends. The post is publishable with these three fixes and the inherited Part 1 caveats carried forward; without them, it reads as engineering reasoning shaped to a narrative rather than the other way around.
