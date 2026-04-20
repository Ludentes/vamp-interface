# Demographic Classifiers for Flux Conditioning Orthogonalization

Date: 2026-04-20
Context: `docs/research/2026-04-20-ffa-progression-curriculum.md` — we hand-design D-dim geometry in Flux conditioning space and need to project out demographic axes so perception discriminations are not trivially solved via "older / younger" or "male / female" cues. Plan: sample ~500 Flux Krea portraits, run each through FairFace + DEX + MiVOLO, regress predicted labels on the Flux conditioning vector, extract top predictive directions per classifier, union (or PCA of stacked direction matrix) as the demographic subspace.

This doc documents what each classifier actually is, what its label schema means, and — the part that matters most — the caveats for interpreting the orthogonalization result.

## FairFace — Kärkkäinen & Joo, WACV 2021

Primary source: Kärkkäinen & Joo, "FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation," WACV 2021 (arXiv 1908.04913). Code: https://github.com/dchen236/FairFace.

**What it outputs.** Three softmax heads:

- Race, two variants: `race_7` over {White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern}, and `race_4` over {White, Black, Asian, Indian}.
- Gender over {Male, Female} — binary, no non-binary category.
- Age over 9 bins: {0–2, 3–9, 10–19, 20–29, 30–39, 40–49, 50–59, 60–69, 70+}.

Output is confidence scores per class, not hard labels. Input is dlib-cropped and aligned 224×224 face; the repo's `predict.py` runs dlib CNN face detection first. Backbone is ResNet-34 pretrained on ImageNet, fine-tuned on FairFace.

**Training corpus.** 108,501 images scraped from YFCC-100M Flickr, labeled by Amazon Mechanical Turk for race/gender/age, deliberately balanced across the seven race groups. The headline claim of the paper is that models trained on FairFace generalize more consistently across race subgroups than models trained on CelebA / UTKFace / LFWA+, which are overwhelmingly White.

**Known failure modes & bias caveats.**

- The 7-way race taxonomy is a contested social construct, not a biological truth; "Middle Eastern" and "Latino_Hispanic" in particular overlap visually with "White" in ways that depend on photographer, lighting, and within-group variation. The paper itself frames this as "race as perceived by annotators."
- Age bins are coarse, especially the 20-year bin from 50–69 collapsed into two wide buckets vs. fine splits at young ages. Errors are largest at the young end (0–2 vs 3–9 often conflated) and at the old end (everything over 60 compresses).
- Gender is binary by construction; any gender-ambiguous Flux output is forced onto the Male/Female axis.
- Annotations come from AMT crowd workers, so the labels FairFace learned are annotator perception, not ground truth.

**License.** The code repo has no explicit license file; the dataset is released for non-commercial research use per the paper. Treat as research-only.

**Install story.** No PyPI package. Clone repo, download weight .zip from a Google Drive link referenced in the README, install `torch`, `torchvision`, `dlib`, `pandas`. dlib needs a C++ toolchain. Works from CSV of image paths → CSV of scores.

## DEX — Rothe, Timofte, Van Gool, IJCV 2018 (ICCV-W 2015)

Primary sources: Rothe, Timofte, Van Gool, "DEX: Deep EXpectation of apparent age from a single image," ICCV Workshop 2015; and the IJCV 2018 extension "Deep expectation of real and apparent age from a single image without facial landmarks." Project page: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/.

**What it outputs.** A distribution over 101 age classes (0–100), collapsed into a single scalar via softmax-expectation: `age_pred = Σ i · softmax(logits)_i`. This is continuous in value but a weighted sum of discrete class probabilities. The IJCV paper also reports gender (binary) but the age model is the canonical "DEX."

**Training corpus.** IMDB-WIKI — 523,051 celebrity images scraped from IMDb and Wikipedia profile pages (460,723 IMDb + 62,328 Wikipedia). Real age derived from `photo_taken` year minus `date_of_birth`; authors explicitly note accuracy is not guaranteed (many stills are from movies with long production timelines). Final training uses an equalized age distribution. Backbone is VGG-16 pretrained on ImageNet, with an ensemble of 20 networks at test time for the ICCV submission.

**Known failure modes.**

- Celebrity-bias: IMDb actors skew younger-looking, more made-up, more professionally-lit than the general population. Predictions drift young on non-celebrity faces.
- Children are heavily underrepresented in IMDB-WIKI (actors with IMDb pages skew adult).
- "Apparent age" is explicitly the training target for the ChaLearn challenge version; the IJCV extension adds real-age models. Which weight file you load matters — apparent-age DEX will give systematically different numbers than real-age DEX.
- VGG-16 is old and fragile to non-frontal poses and heavy occlusion.
- The softmax-expectation smooths output but does not give a real uncertainty estimate. Bimodal posteriors (e.g. "looks 10 OR 60") collapse to the mean (35).

**License.** Dataset license: non-commercial research only, per the project page ToS. Weights originally released in MatConvNet and Caffe; several PyTorch ports exist (e.g. `siriusdemon/pytorch-DEX`) but none are official.

**Install story.** No official PyPI. You pick a community port (PyTorch re-implementations of VGG-16 loading Caffe weights). Input expected to be face-cropped ~224×224. No built-in face detector in the canonical weights — you supply a crop.

## MiVOLO — Kuprashevich & Tolstykh, arXiv 2307.04616 (2023)

Primary sources: Kuprashevich & Tolstykh, "MiVOLO: Multi-input Transformer for Age and Gender Estimation," arXiv 2307.04616 (2023); follow-up Kuprashevich, Alekseenko, Tolstykh, arXiv 2403.02302 (2024). Code: https://github.com/WildChlamydia/MiVOLO.

**What it outputs.** Joint age (continuous regression) + gender (binary) head. Two input branches: a face crop and a person-body crop, fed to a VOLO-D1 vision transformer backbone with a dual-input fusion. If the face is missing, it can use body-only; if body is missing, face-only.

**Training corpus & benchmarks.** Authors introduce **Lagenda**, a new annotated dataset curated from Open Images, with votes aggregated across multiple annotators. Reported numbers from the repo:

- IMDB-cleaned: age MAE 4.22 (face-only, VOLO-D1); 4.24 face+body (MiVOLO-D1); gender 99.38–99.46%.
- Lagenda train → FairFace val: age accuracy 61.07% face+body, gender 95.73%.
- Lagenda train → Adience: age 68.69–69.43%, gender 96.51–97.39%.
- Body-only age MAE 6.87 — substantially worse than face+body.
- Claim of "beats human annotators on most age ranges" is from a human-vs-model study they ran themselves on their Open Images benchmark.

**License.** Apache-2.0 (code). Weights: research-use framing in README; check each checkpoint's card.

**Portrait-only degradation.** This is the one we need to care about. Our Flux samples are head-and-shoulders portraits without a full body crop. The MiVOLO-D1 checkpoint accepts face-only inference (the code handles missing body), and the repo shows face-only MAE on AgeDB of 5.55–5.58 — worse than full-person IMDB-cleaned (4.24) but still usable. Face-only means you lose the multi-input advantage MiVOLO was built for; you are effectively running a VOLO-D1 face classifier with heavier weights and marginal gains over FairFace/DEX. The repo also publishes a face-only `volo_d1` checkpoint that skips the body branch entirely, which is probably the cleaner choice for our pipeline.

**Known failure modes.**

- Gender is still binary.
- No race/ethnicity head at all — MiVOLO contributes only age and gender directions.
- Trained heavily on Open Images and IMDB-cleaned, which are Western-media-skewed.
- Transformer models are sensitive to input resolution; default is 224 or 384 depending on checkpoint.

**Install story.** No PyPI. Clone repo, `pip install -r requirements.txt` (torch, timm, a specific `ultralytics` for the bundled detector). Weights on HuggingFace Hub. The detector is a YOLO variant they package — you can bypass it and feed your own face crops. Some pinned dependency versions in `requirements.txt` may conflict with current torch; expect to relax pins.

## Cross-cutting caveats for our orthogonalization plan

### How the three classifiers disagree, and what to do about it

- **Age representation mismatch.** FairFace gives a categorical over 9 bins. DEX gives a softmax-expectation scalar over 101 classes. MiVOLO gives a continuous regression. If we regress conditioning → predicted-age for each, the three predictive directions will not coincide. FairFace's direction will be pulled toward whatever discriminates the bin boundaries (especially the coarse elderly bins); DEX's will be pulled toward celebrity-looking photos; MiVOLO's toward Open-Images-style framing. **Recommendation:** do not average. Stack the three direction sets and take union (or run PCA over the stacked matrix) — taking union is more conservative (projects out more) but is the right default when disagreement itself is evidence of surface-attribute leakage.
- **Race only comes from FairFace.** DEX has no race head. MiVOLO has no race head. So racial directions will be single-source. That is a concentration risk — FairFace's race boundaries become the only signal, and any FairFace idiosyncrasy (e.g. its White/Middle-Eastern confusion on olive-skinned Flux outputs) becomes a load-bearing artifact of our orthogonalization.
- **Gender is voted 3×.** All three predict gender, all three predict binary. That is good for robustness on the gender axis, but it does not rescue us from the "no non-binary" problem. Any gender-ambiguous Flux output contributes noisy labels, and the regression direction becomes the mean of three noisy rulers.
- **Body-aware vs face-only.** MiVOLO's key claim is multi-input. We strip that. Our MiVOLO predictions are effectively face-only VOLO-D1 — well-regularized, but not the model that earned the SOTA numbers.

### Real-photo-trained → synthetic-Flux-evaluated

All three classifiers were trained on real photographs. We are applying them to generative-model output. There is, to our knowledge, **no published validation** that FairFace, DEX, or MiVOLO transfer sensibly to Flux or any other rectified-flow / diffusion portrait generator. Known related findings:

- CLIP-based face classifiers degrade on GAN outputs in ways that correlate with GAN artifacts rather than ground-truth attributes (Chai et al., "What makes fake images detectable," ECCV 2020, adjacent finding).
- Flux Krea specifically produces anatomically crisp faces with fewer obvious GAN tells than StyleGAN-era output, which likely *reduces* transfer gap but does not eliminate it.
- **Assumption to flag in any downstream claim:** "classifier predictions on Flux outputs correlate monotonically with what a human would call the corresponding demographic." We have not validated this.

A minimal sanity check before trusting the orthogonalization: take 50 Flux outputs, have a human label age-bin and gender, compare to each classifier. If any classifier has <70% agreement with human labels on Flux outputs, its directions are suspect.

### Label-as-target vs identity-as-target

We are regressing on **predicted** labels, not ground-truth labels. The directions we extract therefore encode "what FairFace/DEX/MiVOLO think is the demographic axis" in our Flux sample. This means:

- Systematic classifier errors become part of the orthogonalized subspace. If FairFace systematically calls dark-haired Flux women "East Asian" regardless of other features, the "East Asian" direction becomes partially a dark-hair direction.
- Any feature Flux uses to encode perceived-age that also correlates with something else (lighting, lens blur, skin texture noise) gets packed into the age direction.
- This is not strictly a bug. For our goal — "remove cues that make perception tasks trivial" — what matters is whether the projected-out subspace captures the cues observers would use. Classifier-perceived demographics are probably a reasonable proxy for observer-perceived demographics, since they were trained on human annotations.
- But **it is absolutely not safe to call the orthogonalized subspace "demographic-free in a fairness sense."** It is classifier-label-free, which is weaker and narrower. If we ever publish, this distinction must be in the abstract.

### Regression-direction vs PCA-of-labels

The plan says regression (predict label from conditioning, extract predictive direction). This is defensible and better than naive PCA-of-label-vectors for these reasons:

- PCA of (age, gender_probs, race_probs) over 500 samples gives directions in the **label space**, not the conditioning space. To project out of the conditioning space, you still need a regression or CCA step to map label-space directions back. Naive PCA on concatenated label vectors does not give you anything you can project with.
- Linear regression of each label onto conditioning gives a direction per label in the 768-d (or whatever-d) conditioning space directly — ready to orthogonalize against.
- For categorical labels, use multinomial logistic regression (or one-vs-rest). The logit gradient direction in conditioning space is the natural "this direction makes the classifier more confident of class k" axis.
- For continuous labels (DEX age, MiVOLO age), use ridge regression. The weight vector is the direction.
- After extracting ~15 directions total (9 age bins × 1 + 1 DEX + 1 MiVOLO + 7 race + 2 gender = ~20, minus redundancy), stack into a matrix and take a truncated SVD. The top-k right singular vectors span the demographic subspace; project conditioning onto the orthogonal complement.

Note: some of these directions will be near-parallel (e.g. FairFace age-bin-30s and DEX continuous age). The SVD handles that gracefully — redundant directions collapse into the same singular vector.

### Race in generative models: what does "Southeast Asian" on a Flux output even mean

FairFace's 7-way race taxonomy was designed as a best-effort balance, not a ground truth. On a Flux output, the label is **doubly fictional**:

- The portrait itself is synthetic — there is no real person whose race could be ground-truthed.
- The classifier was trained on human annotators' perception of real photographs, and the model has learned a decision boundary that conflates skin tone, hair texture, facial geometry, and photographic style.

Practical consequence: our "race subspace" in Flux conditioning space is roughly "the direction that moves skin tone, hair texture, and a few geometric features together in a way that FairFace recognizes." That is fine for removing cues that would let a perception task be solved via "different-looking skin," but it is **not** a race variable in any measurable sense. It is a visual-category variable with a social label attached. Be careful not to overclaim in writeups.

## Open questions / assumptions to test empirically

- [ ] On 50 Flux outputs, does each classifier agree with human-labeled age-bin and gender at >70%? If not, drop or reweight that classifier.
- [ ] After orthogonalization, sample random points in the orthogonal complement, render, and run the classifiers again. Do predictions become near-chance / low-confidence? If they still predict age and gender strongly, our subspace is too narrow — probably missing nonlinear age cues (wrinkles, eye shape) that linear regression cannot capture.
- [ ] Does union-of-directions vs SVD-of-stacked-matrix make a visible difference on held-out samples? If yes, SVD is probably over-projecting (removing content we care about); union may be safer.
- [ ] Does MiVOLO face-only add anything over FairFace age-bins + DEX? If the three age predictions are all >0.9 correlated on our sample, MiVOLO is redundant and we can drop it.
- [ ] Is 500 Flux samples enough? Rule-of-thumb for ridge regression is ~10× the number of directions you want, per output label. With ~20 labels and a 768-d conditioning space, 500 is thin. Consider 1500–2000.
- [ ] Nonlinear demographic cues: if linear orthogonalization leaves visible age/gender signal, test a small MLP-regression and project out its input Jacobian directions evaluated at sample points. This is adversarial-robustness territory and expensive; only do it if the linear version fails.
