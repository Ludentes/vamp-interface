# ARKit→FLAME retargeter — deferred from X6 bake-off

The bake-off uses LAM's **bundled sample motion sequences** (pre-extracted FLAME params under
`assets/sample_motion/export/<name>/flame_params.json`), not our LLF take. LAM's inference path
does not consume video drivers natively — it consumes pre-baked FLAME parameter sequences.
Extracting FLAME from a custom video requires running VHAP (vendored in `LAM/vhap/`) on the
driver clip, which is a non-trivial sub-pipeline. Deferred.

This isolates the question we actually care about for X6 — does LAM's identity-encoder + GS renderer
hold up on painted/stylized anchors — from the secondary question of whether our live ARKit stream
maps cleanly into FLAME.

If LAM passes the bake-off, the follow-on "open VASA stack" project needs two steps to deliver
the live-streaming use case (no recorded driver video):

1. **Video → FLAME (offline test)**: Run VHAP on our LLF takes to confirm the motion-extraction
   path produces clean FLAME sequences from iPhone face-cam input. Should work out of the box,
   VHAP is LAM's training-time tracker.

2. **Live ARKit → FLAME (online)**: Build a direct ARKit-blendshapes → FLAME-expression map.
   Candidates ranked by community traction:

   - **MICA / EMICA** — strongest open ARKit-to-FLAME pipeline; FLAME expression decoders with
     ARKit-compatible heads. https://github.com/Zielon/MICA
   - **SMIRK** — recent, lightweight, more focused on expression-only.
     https://github.com/georgeretsi/smirk
   - **INSTA** — Zielonka et al, includes an ARKit-to-FLAME regression step in their preprocessing.
   - **Custom regression** — fit a 52-dim ARKit → (100-dim FLAME-exp + jaw + eye) linear map on a
     paired dataset (re-extract FLAME from our existing LLF-take videos using MICA, pair with the
     LLF ARKit stream, ridge-fit the map). ~1-day project once LAM is the renderer of record.

None of these are X6 blockers.
