# Report: 2026 papers (Jan–Apr) on neural portrait/talking‑head animation related to LivePortrait / First‑Order motion‑transfer lineage

Executive summary  
This report compiles 2026 papers (arXiv postings with YYMM 2601–2604 and conference items where available in the provided evidence) that are directly about or plausibly relevant to neural portrait animation, image deformation, implicit keypoint motion transfer, or closely related methods. Each paper entry shows provenance, a concise technical summary drawn only from the provided evidence, code availability as reported in the evidence, and whether the work explicitly connects to the LivePortrait / First‑Order lineage (or is only possibly relevant). The report also verifies four previously‑claimed items (PersonaLive, FantasyTalking2, UniTalking, DyStream) against the available evidence and summarizes gaps and uncertainties. All factual statements are supported by the listed source URLs in the References.

## 1. Focused list of 2026 (2601–2604) papers and closely related items

Each entry: Title; authors (only when present in the evidence); arXiv ID (if any); venue; 1–2 sentence technical contribution (evidence‑based); code availability (per evidence) + link if present; connection to LivePortrait / First‑Order lineage (explicit or judged from facts). Provenance and whether inside cutoff are shown.

1) Splat‑Portrait: Accurate 3D Splat Representation for Portrait Animation  
- arXiv ID / provenance: arXiv 2601.18633 (inside cutoff) [1].  
- Venue: arXiv preprint (2601.18633v1) [1].  
- Summary: The method disentangles a single portrait image into a 3D Gaussian splat representation of the head with an inpainted 2D background and animates 3D splats from audio and time deltas by predicting dynamic offsets without an explicit deformation model; it is trained self‑supervised from monocular videos and distills a strong 2D face prior (no 3D supervision) [1].  
- Code: Not reported in the provided evidence.  
- Connection to LivePortrait / First‑Order: Possibly relevant — it explicitly models dynamic per‑pixel/point offsets and a view‑space splat representation rather than 2D keypoint motion transfer; the evidence does not state it builds on LivePortrait or First‑Order [1].  
- Evidence: arXiv page (HTML) [1].

2) From Blurry to Believable / SuperHead: Enhancing Low‑quality Talking Heads with 3D Generative Priors  
- arXiv ID / provenance: arXiv 2602.06122 (inside cutoff); accepted to 3DV 2026 per evidence [2].  
- Venue: arXiv preprint; reported acceptance to 3DV 2026 [2].  
- Summary: Introduces a dynamics‑aware 3D inversion scheme that optimizes latents of a generative model to produce a super‑resolved 3D Gaussian Splatting (3DGS) head model rigged to a parametric head (e.g., FLAME) for animation, using upscaled 2D render and depth supervision; aims to enhance low‑resolution animatable 3D head avatars [2], [11].  
- Code: Public GitHub repository for "SuperHead" is present per the evidence [11].  
- Connection to LivePortrait / First‑Order: Relevant to talking‑head/portrait animation and to explicit 3D avatar representations; the evidence does not assert a direct lineage connection to LivePortrait or First‑Order [2], [11].  
- Evidence: arXiv abstract [2] and GitHub repo [11].

3) FlexiMMT: Implicit Multi‑Object Multi‑Motion Transfer (CVPR 2026)  
- arXiv ID / provenance: arXiv 2603.01000; reported accepted to CVPR 2026 (inside cutoff / CVPR 2026) [3].  
- Venue: arXiv preprint and reported CVPR 2026 acceptance [3].  
- Summary: Proposes multi‑object, multi‑motion transfer from a static image using Motion Decoupled Mask Attention to constrain attention per object and Differentiated Mask Propagation to derive and propagate object masks across frames; targets compositional video generation and assigns independent motion patterns to distinct objects [3], [9].  
- Code: Public GitHub repository available per evidence [9].  
- Connection to LivePortrait / First‑Order: Possibly relevant in method family (motion transfer) but the described focus is multi‑object compositional video rather than face/talking‑head explicit implicit‑keypoint motion transfer; no explicit claim of First‑Order / LivePortrait lineage in the provided evidence [3], [9].  
- Evidence: arXiv abstract [3] and GitHub [9].

4) Motion Manipulation via Unsupervised Keypoint Positioning in Face Animation (MMFA)  
- arXiv ID / provenance: arXiv 2603.04302 (inside cutoff) [4].  
- Venue: arXiv preprint (2603.04302) [4].  
- Summary: MMFA represents faces using 3D keypoints to enable arbitrary pose alterations (rotation/translation) and supports motion attribute editing in face animation; the method couples expression deformation with facial scaling which limits accurate expression manipulation according to the evidence [4].  
- Code: Not reported in the provided evidence.  
- Connection to LivePortrait / First‑Order: Directly relevant — uses an (unsupervised) keypoint representation for face animation, which places it in the implicit‑keypoint / keypoint motion‑transfer family related to First‑Order style methods; the evidence does not state explicit citation of LivePortrait/First‑Order but the approach matches the lineage conceptually [4].  
- Evidence: arXiv abstract/html [4].

5) IUP‑Pose: Decoupled Iterative Uncertainty Propagation for Real‑time Relative Pose Regression via Implicit Dense Alignment  
- arXiv ID / provenance: arXiv 2603.19625 (inside cutoff) [5].  
- Venue: arXiv preprint (2603.19625v1) [5].  
- Summary: Uses spatial pyramid pooling as an implicit keypoint detection technique and avoids explicit keypoint detection by performing feature alignment through a multi‑head bidirectional cross‑attention layer, reducing computational cost by operating at stride 32 for the IDA module [5].  
- Code: Not reported in the provided evidence.  
- Connection to LivePortrait / First‑Order: Methodologically relevant because it explicitly uses “implicit keypoint” style detection/alignment mechanisms; however, the evidence frames the work as relative pose regression rather than portrait animation, so it is a potentially useful technical follow‑up rather than a direct face animation successor [5].  
- Evidence: arXiv HTML [5].

6) Neural Image Space Tessellation (post‑processing for silhouettes)  
- arXiv ID / provenance: arXiv 2602.23754 (inside cutoff) [7].  
- Venue: arXiv preprint (2602.23754) [7].  
- Summary: A post‑processing approach inspired by Phong tessellation that refines silhouettes in screen space to remove silhouette artifacts and preserve high‑frequency appearance while producing tessellation‑like visual effects from low‑poly geometry [7].  
- Code: Not reported in the provided evidence.  
- Connection to LivePortrait / First‑Order: Possibly relevant as a post‑processing/rendering refinement that could be applied to portrait animation outputs; no direct lineage claim in the evidence [7].  
- Evidence: arXiv abstract [7].

7) 6D Object Pose + Neural Scene Representation (joint optimization)  
- arXiv ID / provenance: arXiv 2604.06720 (inside cutoff) [6].  
- Venue: arXiv preprint (2604.06720v1) [6].  
- Summary: Jointly optimizes camera pose and a neural scene representation mapping world coordinates to color and TSDF with multi‑loss objectives and introduces a mask alignment loss; sampling strategies and use of canonical meshes are described in the evidence [6].  
- Code: Not reported in the provided evidence.  
- Connection to LivePortrait / First‑Order: Borderline relevance — the work focuses on object pose and neural scene representation, not explicitly on portrait/talking‑head animation or implicit keypoint motion transfer in face animation per provided facts [6].  
- Evidence: arXiv HTML [6].

8) UniTalking: A Unified Audio‑Video Framework for Talking Portrait Generation (arXiv 2603.01418) — noted for verification below  
- arXiv ID / provenance: arXiv 2603.01418 (inside cutoff); arXiv lists it as accepted to CVPR 2026 Findings per the evidence [13], [14].  
- Venue: arXiv; listed on arXiv as CVPR 2026 (Findings) in the provided evidence [13], [14].  
- Summary: Uses Multi‑Modal Transformer Blocks to jointly model audio and video latent tokens for synchronized talking‑portrait generation and reports improved Sync‑C scores relative to baseline models in the evidence [13], [14].  
- Code: Not reported in the provided evidence.  
- Connection to LivePortrait / First‑Order: Relevant to talking‑portrait generation; the evidence does not specify whether it extends explicit implicit‑keypoint motion‑transfer or First‑Order lineage [13], [14].  
- Evidence: arXiv abstract/html [13], [14].

Notes on scope: all the entries above are included only when supported by the provided evidence; items outside the 2601–2604 cutoff are addressed separately below as “outside cutoff” follow‑ups.

## 2. Verification of four previously‑claimed papers (labels + evidence)

For each claimed item the required checks were: (a) arXiv metadata & PDF existence and match to title, (b) whether paper text/abstract matches the claimed topic (talking‑head / streaming / motion transfer), and (c) whether claimed conference acceptance (CVPR/AAAI/etc.) is supported by official program/proceedings. Evidence links are given for the arXiv pages and for program/proceedings absence where noted.

1) PersonaLive — arXiv:2512.11253 “PersonaLive! Expressive Portrait Image Animation for Live Streaming”  
- ArXiv metadata & PDF: The arXiv entry exists as arXiv:2512.11253 with that title per the evidence [12].  
- Topic match: The project claims a real‑time, streamable diffusion framework for infinite‑length portrait animations and WebUI live streaming functionality per the project site referenced in the evidence [12], [13].  
- Claimed venue acceptance (CVPR 2026): No supporting evidence of CVPR 2026 acceptance was found in the provided CVPR 2026 program page search; the provided evidence explicitly states no official CVPR 2026 program page was found confirming acceptance of PersonaLive (or FantasyTalking2 or DyStream) [12], [19].  
- Label: Contradicted — the arXiv entry and project pages exist and match the title and claimed talking‑head/streaming topic, but there is no evidence in the provided sources that PersonaLive was accepted to CVPR 2026; the CVPR 2026 program page search did not confirm its acceptance [12], [13], [19].  
- Evidence links: arXiv entry [12]; project site / demo page [13]; CVPR 2026 virtual/papers page where no confirmation was found [19].

2) FantasyTalking2 — arXiv:2508.11255 (claimed AAAI 2026 acceptance)  
- ArXiv metadata & PDF: The arXiv entry exists as arXiv:2508.11255 and PDF is available per the evidence [12].  
- Topic match: The arXiv abstract describes timestep‑layer adaptive preference optimization for audio‑driven portrait animation, consistent with the claimed topic [12].  
- Claimed venue acceptance (AAAI 2026): No supporting evidence of AAAI 2026 acceptance was found in the provided AAAI conference proceedings page search; the provided evidence notes no official AAAI 2026 program page confirmation for FantasyTalking2 [12], [20].  
- Label: Contradicted — arXiv and PDF exist and match the described audio‑driven portrait animation topic, but there is no evidence in the provided sources that FantasyTalking2 was accepted to AAAI 2026 [12], [20].  
- Evidence links: arXiv abstract and PDF [12]; AAAI proceedings page where no confirmation was found [20].

3) UniTalking — arXiv:2603.01418 (claimed CVPR 2026 Findings)  
- ArXiv metadata & PDF: ArXiv entry arXiv:2603.01418 exists and the HTML abstract/page lists it as "accepted to CVPR 2026 (Findings)" per the evidence [13], [14].  
- Topic match: The arXiv abstract describes a unified audio‑video transformer framework for talking‑portrait generation, with reported Sync‑C improvements and comparisons to other models, matching the talking‑head topic [13], [14].  
- Claimed venue acceptance (CVPR 2026 Findings): The arXiv metadata itself lists CVPR 2026 (Findings) acceptance [13]. However, the provided search of the official CVPR 2026 program pages (the available CVPR 2026 virtual/papers page) did not surface a separate independent confirmation for the set of claims noted elsewhere (the provided evidence explicitly states there was no found confirmation for some other titles, but does not include an official CVPR program page confirming UniTalking beyond the arXiv claim) [19].  
- Label: Uncertain — arXiv metadata explicitly lists CVPR 2026 (Findings) acceptance, and the paper content matches talking‑portrait generation, but an independent confirmation from an official CVPR 2026 program/proceedings page is not present in the provided evidence; thus acceptance cannot be fully independently corroborated from the sources given [13], [14], [19].  
- Evidence links: arXiv abstract/html [13], [14]; note on absence of explicit CVPR 2026 program confirmation in the provided CVPR page search [19].

4) DyStream — arXiv:2512.24408 “DyStream: Streaming Dyadic Talking Heads Generation via Flow Matching‑based Autoregressive Model”  
- ArXiv metadata & PDF: The arXiv entry exists as arXiv:2512.24408 and includes HTML content showing submission on 30 Dec 2025 and revision on 2 Feb 2026 with the stated title and abstract describing a streaming flow‑matching autoregressive model for dyadic talking‑head generation [12], [17].  
- Topic match: The arXiv abstract describes a flow‑matching autoregressive model for dyadic talking‑head video, single reference image input, dual‑stream audio, and claims of low latency and strong lip‑sync metrics, which matches the claimed streaming/talking‑head topic [12], [17].  
- Claimed venue acceptance: No evidence in the provided sources indicates formal acceptance to a named conference or presence in official proceedings; the provided CVPR 2026 search did not confirm acceptance of DyStream among the items explicitly checked [19]. The project page (author page) exists and documents the system but does not constitute official conference acceptance evidence in the provided citations [18].  
- Label: Uncertain — arXiv record and project page confirm the paper and its streaming/dyadic talking‑head topic, but there is no supporting evidence in the provided sources that it was accepted to a particular conference or proceedings [12], [17], [18], [19].  
- Evidence links: arXiv abstract/html [17]; project page [18]; CVPR program search absence [19].

## 3. Follow‑up families (Hallo3 / Hallo4 / FLOAT / EMO) — presence and cutoff status

- Hallo3 (arXiv 2412.00733) — outside cutoff. Evidence describes Hallo3 as "Highly Dynamic and Realistic Portrait Image Animation with Video Diffusion Transformer" and shows a CVPR 2025 paper PDF and arXiv entry; this is therefore a prior (2024/2025) work in the same general portrait animation space but outside the 2601–2604 cutoff [16], [17].  
- Hallo4 (arXiv 2505.23525) — outside cutoff. Evidence indicates Hallo4 focuses on direct preference optimization and temporal motion modulation to improve lip‑sync and expression naturalness; outside the 2601–2604 cutoff [18], [21].  
- FLOAT (arXiv 2412.01064) — outside cutoff. Evidence indicates FLOAT uses generative motion latent flow matching for audio‑driven talking portraits, with transformer‑based vector‑field predictor and flow‑matching in motion‑latent space; outside the 2601–2604 cutoff [22].  
- EMO follow‑ups: No EMO‑followup papers are present in the provided evidence.  
- Conclusion on follow‑ups: Hallo3/Hallo4/FLOAT are confirmed in the evidence but are explicitly outside the 2601–2604 arXiv cutoff and are therefore reported as "outside cutoff" follow‑ups per the brief [16], [18], [22].

## 4. Notes on provenance, inclusion criteria, and inside/outside cutoff labels

- All included 2026 arXiv items with IDs starting 2601–2604 are labeled "inside cutoff" and are supported by the corresponding arXiv HTML/abstract pages in the evidence (e.g., 2601.18633, 2602.06122, 2602.23754, 2603.01000, 2603.01418, 2603.04302, 2603.19625, 2604.06720) [1], [2], [7], [3], [13], [4], [5], [6].  
- Items reported as accepted to conferences in the evidence are labeled with that venue (e.g., From Blurry to Believable / SuperHead reported accepted to 3DV 2026 [2]; FlexiMMT reported accepted to CVPR 2026 per the arXiv entry [3]). Where the evidence lists a conference acceptance on arXiv but no independent program/proceedings confirmation is present in the provided sources, that acceptance is treated as uncorroborated and noted in the Verification section (see UniTalking entry and verification) [13], [14], [19].  
- Papers and project pages outside the 2601–2604 cutoff (PersonaLive 2512.11253, FantasyTalking2 2508.11255, DyStream 2512.24408, Hallo3/4, FLOAT) are reported as outside cutoff and listed under the verification and follow‑up sections with explicit labels and evidence [12], [17], [16], [18], [21], [22].

## 5. Evidence gaps and uncertainties

- Conference acceptance corroboration: For several claims of acceptance to major conferences (notably PersonaLive, FantasyTalking2, and DyStream) the provided evidence contains the arXiv entries and project pages but lacks an independent official conference program/proceedings page confirming acceptance; therefore acceptance claims are either contradicted (when the claim was explicitly stated but no evidence found) or remain uncertain pending authoritative program/proceedings confirmation [12], [19], [20].  
- Code availability: For a number of 2601–2604 arXiv items the evidence does not report code locations; only FlexiMMT and SuperHead have GitHub repositories in the provided evidence [9], [11]. Absence of a code link in the evidence should not be read as absence of code generally, only absence in the provided findings.  
- Direct LivePortrait / First‑Order lineage statements: The provided evidence does not contain explicit citation chains tying many of the 2601–2604 papers to LivePortrait or First‑Order Motion Model; for those that use unsupervised/implicit keypoints or flow‑matching style techniques (e.g., MMFA, IUP‑Pose, DyStream's flow‑matching claim) the connection is described conservatively as "conceptually related" or "possibly relevant" unless the evidence explicitly states a direct extension [4], [5], [17].

## 6. Short synthesis and actionable takeaways

- Within the 2601–2604 arXiv window, several papers advance portrait/talking‑head and motion‑transfer directions with techniques that are relevant to the First‑Order/LivePortrait family: notably MMFA (unsupervised keypoint positioning for face animation), IUP‑Pose (implicit keypoint detection / implicit dense alignment), Splat‑Portrait (3D Gaussian splat representation animated from audio), and UniTalking (audio‑video unified transformer for talking portraits) [1], [4], [5], [13], [14].  
- Flow‑matching approaches remain present in the domain (DyStream claims a flow‑matching autoregressive model for streaming dyadic talking heads, and FLOAT remains a prior flow‑matching motion‑latent approach outside the cutoff) — this suggests continued interest in flow‑matching and latent motion fields for low‑latency and high‑fidelity lip sync / motion transfer [17], [22].  
- For reproducibility and follow‑up work, the evidence shows public code for FlexiMMT and SuperHead only; missing code links for several other 26xx items imply verification or implementation effort will be needed to reproduce those methods from the papers alone [9], [11].  
- Official conference acceptance status for some arXiv papers that claim or are reported as accepted (notably UniTalking and FlexiMMT) is not fully independently corroborated in the provided CVPR program search evidence for all items, so users relying on conference acceptance as a signal of peer review should consider confirming via the official conference proceedings pages or program lists beyond the provided evidence [13], [14], [19].

## References (numbered sources)

[1] https://arxiv.org/html/2601.18633v1  
[2] https://arxiv.org/abs/2602.06122  
[3] https://arxiv.org/abs/2603.01000  
[4] https://arxiv.org/abs/2603.04302  
[5] https://arxiv.org/html/2603.19625v1  
[6] https://arxiv.org/html/2604.06720v1  
[7] https://arxiv.org/abs/2602.23754  
[8] https://github.com/Ethan-Li123/FlexiMMT  
[9] https://github.com/humansensinglab/super-head  
[10] https://arxiv.org/abs/2512.11253  
[11] https://personalive.app/  
[12] https://arxiv.org/abs/2508.11255  
[13] https://arxiv.org/abs/2603.01418  
[14] https://arxiv.org/html/2603.01418v1  
[15] https://arxiv.org/abs/2512.24408  
[16] https://arxiv.org/abs/2412.00733  
[17] https://openaccess.thecvf.com/content/CVPR2025/papers/Cui_Hallo3_Highly_Dynamic_and_Realistic_Portrait_Image_Animation_with_Video_CVPR_2025_paper.pdf  
[18] https://arxiv.org/abs/2505.23525  
[19] https://cvpr.thecvf.com/virtual/2026/papers.html  
[20] https://aaai.org/aaai-publications/aaai-conference-proceedings/  
[21] https://arxiv.org/html/2505.23525v1  
[22] https://arxiv.org/html/2412.01064v5  
[23] https://papers.cool/arxiv/2602.06122  
[24] https://catalyzex.com/author/Xinyu%20Zhang  
[25] https://linkedin.com/posts/jonathansilvasantos_videogeneration-computervision-generativeai-activity-7434817926509006849-lU0Z  
[26] https://runcomfy.com/comfyui-nodes/ComfyUI-PersonaLive  
[27] https://comfy.icu/extension/okdalto__ComfyUI-PersonaLive  
[28] https://arxiv.org/pdf/2508.11255  
[29] https://arxiv.org/html/2512.24408v1  
[30] https://robinwitch.github.io/DyStream-Page/

(Each numbered References entry above corresponds to a unique URL that was used as the evidence basis for the statements in this report. Citations in the body point to these numbered sources.)