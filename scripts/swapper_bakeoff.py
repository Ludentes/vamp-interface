"""Swap-stage bake-off: inswapper_128 vs hyperswap_1a/1b_256.

Isolates the face-swapper. Reuses the best-cell doll renders from the CN grid
sweep (strength 0.90 / 6 steps) as fixed targets, and swaps each of the 20
importer identities onto its own doll with every backend. The detect / crop /
upscale / collapse / feathered paste-back path (swap_core.swap_identity) is
held constant -- only the swapper object changes.

Each backend exposes the InsightFace INSwapper signature
`get(img, target_face, source_face, paste_back=True)`, so swap_identity runs
unmodified. inswapper_128 loads via the stock INSwapper class; HyperSwap is a
faithful port of FaceFusion's hyperswap inference (arcface_128 warp template,
[-1,1] normalisation, L2-normed source embedding, model-emitted face mask).

ReSwapper-256 was tested and dropped: on the small painted doll face it is
well-aligned but does not carry identity (ArcFace cos ~0.2 on the recognition
model, vs ~0.86 for inswapper) across every source-latent convention -- not an
integration bug, the reimplementation is simply not strong enough here.

Resumable: one JSONL row per (backend, identity); completed cells skipped.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

os.chdir("/home/newub/w/vamp-interface")
sys.path.insert(0, "scripts")
from swap_core import (crop_and_upscale, detect_source, load_swapper,
                       make_face_app, mediapipe_kps_bbox, swap_identity)

WEIGHTS = "exp_output/swapper_bakeoff/weights"
OUT = "exp_output/swapper_bakeoff"
RESULTS = f"{OUT}/results.jsonl"
DOLL = "exp_output/cn_grid/renders/{idn}_str90_st6.png"
SRC = "data/importer/identities/{idn}.png"
INSWAPPER = os.path.expanduser("~/w/ComfyUI/models/insightface/inswapper_128.onnx")
IDENTITIES = [f"id_{i:02d}" for i in range(20)]
_CPU = ["CPUExecutionProvider"]

# FaceFusion WARP_TEMPLATE_SET['arcface_128'] -- normalised 5-point template
# (eyeL, eyeR, nose, mouthL, mouthR), scaled by crop size at warp time.
_ARCFACE_128 = np.array([
    [0.36167656, 0.40387734], [0.63696719, 0.40235469],
    [0.50019687, 0.56044219], [0.38710391, 0.72160547],
    [0.61507734, 0.72034453]], dtype=np.float32)


class HyperSwap:
    """FaceFusion hyperswap_*_256 swapper behind the INSwapper.get signature.

    Port of facefusion/processors/modules/face_swapper/core.py: warp the
    target face to a 256 crop via the arcface_128 template, feed it
    [-1,1]-normalised alongside the L2-normalised ArcFace source embedding,
    de-normalise the output, and paste it back through the model's own mask.
    face_swapper_weight is left at its default 0.5 -> embedding balance is a
    no-op, so it is omitted.
    """

    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(onnx_path, providers=_CPU)
        self.size = 256

    def get(self, img, target_face, source_face, paste_back=True):
        kps = np.asarray(target_face.kps, dtype=np.float32)
        tmpl = _ARCFACE_128 * self.size
        affine = cv2.estimateAffinePartial2D(
            kps, tmpl, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
        crop = cv2.warpAffine(img, affine, (self.size, self.size),
                              borderMode=cv2.BORDER_REPLICATE,
                              flags=cv2.INTER_AREA)

        blob = crop[:, :, ::-1].astype(np.float32) / 255.0     # BGR->RGB, 0..1
        blob = (blob - 0.5) / 0.5                               # -> [-1, 1]
        blob = blob.transpose(2, 0, 1)[None]
        src = source_face.normed_embedding.reshape(1, -1).astype(np.float32)

        out, mask = self.sess.run(None, {"source": src, "target": blob})
        out = out[0].transpose(1, 2, 0)                         # CHW -> HWC
        out = np.clip(out * 0.5 + 0.5, 0, 1)[:, :, ::-1] * 255  # RGB->BGR
        out = out.astype(np.float32)
        face_mask = np.clip(mask[0, 0], 0, 1).astype(np.float32)

        if not paste_back:
            return out.astype(np.uint8)

        inv = cv2.invertAffineTransform(affine)
        h, w = img.shape[:2]
        warped = cv2.warpAffine(out, inv, (w, h),
                                borderMode=cv2.BORDER_REPLICATE)
        warped_mask = cv2.warpAffine(face_mask, inv, (w, h))[..., None]
        warped_mask = np.clip(warped_mask, 0, 1)
        blended = warped * warped_mask + img.astype(np.float32) * (1 - warped_mask)
        return np.clip(blended, 0, 255).astype(np.uint8)


def make_backends():
    return {
        "inswapper_128": load_swapper(INSWAPPER),
        "hyperswap_1a_256": HyperSwap(f"{WEIGHTS}/hyperswap_1a_256.onnx"),
        "hyperswap_1b_256": HyperSwap(f"{WEIGHTS}/hyperswap_1b_256.onnx"),
    }


def id_cos(app, bgr, src_emb):
    """Post-swap ArcFace cosine vs the source identity (matches cn_grid)."""
    kps, bbox = mediapipe_kps_bbox(bgr)
    if bbox is None:
        return float("nan")
    up, _ = crop_and_upscale(bgr, bbox)
    if up is None:
        return float("nan")
    faces = app.get(up)
    if faces:
        return float(np.dot(max(faces, key=lambda x: x.det_score)
                            .normed_embedding, src_emb))
    kps_up, bbox_up = mediapipe_kps_bbox(up)
    rec = app.models.get("recognition")
    if kps_up is None or rec is None:
        return float("nan")
    from insightface.app.common import Face
    f = Face(bbox=bbox_up, kps=kps_up, det_score=1.0)
    rec.get(up, f)
    return float(np.dot(f.normed_embedding, src_emb))


def done_keys():
    if not os.path.exists(RESULTS):
        return set()
    keys = set()
    for line in open(RESULTS):
        line = line.strip()
        if line:
            d = json.loads(line)
            keys.add((d["backend"], d["identity"]))
    return keys


def main():
    app = make_face_app()
    backends = make_backends()
    for name in backends:
        os.makedirs(f"{OUT}/swaps/{name}", exist_ok=True)

    done = done_keys()
    cells = [(b, idn) for b in backends for idn in IDENTITIES]
    todo = [c for c in cells if c not in done]
    print(f"{len(cells)} cells, {len(done)} done, {len(todo)} to run",
          flush=True)

    t0 = time.time()
    for i, (bname, idn) in enumerate(todo):
        doll = cv2.imread(DOLL.format(idn=idn))
        src_bgr = cv2.imread(SRC.format(idn=idn))
        src_face = detect_source(app, src_bgr)
        src_emb = src_face.normed_embedding

        ts = time.time()
        res, mode, det = swap_identity(app, backends[bname], doll, src_face)
        swap_s = time.time() - ts
        cv2.imwrite(f"{OUT}/swaps/{bname}/{idn}.png", res)
        cos = id_cos(app, res, src_emb)

        row = {"backend": bname, "identity": idn, "mode": mode,
               "det": round(float(det), 4),
               "id_cos": round(float(cos), 4) if cos == cos else None,
               "swap_s": round(swap_s, 2)}
        with open(RESULTS, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  [{i+1}/{len(todo)}] {bname:17s} {idn}  {mode:8s} "
              f"cos={cos:.3f} {swap_s:.2f}s", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)
    summarize()
    build_collage()


def summarize():
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    print("\nbackend            n  default  id_cos mean   min   max  swap_s")
    by = {}
    for r in rows:
        by.setdefault(r["backend"], []).append(r)
    for b in sorted(by):
        rs = by[b]
        cos = [r["id_cos"] for r in rs if r["id_cos"] is not None]
        ndef = sum(r["mode"] == "default" for r in rs)
        sw = np.mean([r["swap_s"] for r in rs])
        print(f"{b:17s} {len(rs):2d}  {ndef/len(rs)*100:5.0f}%  "
              f"{np.mean(cos):11.3f}  {np.min(cos):.3f}  {np.max(cos):.3f}  "
              f"{sw:5.2f}")


def build_collage():
    from PIL import Image, ImageDraw, ImageFont
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    backends = sorted({r["backend"] for r in rows})
    cell = {(r["backend"], r["identity"]): r for r in rows}
    TILE, LBL = 200, 24
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)

    def tile(bgr, label):
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        im.thumbnail((TILE, TILE))
        c = Image.new("RGB", (TILE, TILE + LBL), (255, 255, 255))
        c.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2))
        ImageDraw.Draw(c).text((3, TILE + 3), label, fill=(200, 0, 0),
                               font=font)
        return c

    ncol = 1 + len(backends)
    out_rows = []
    for idn in IDENTITIES:
        src = cv2.imread(SRC.format(idn=idn))
        row_img = Image.new("RGB", (TILE * ncol, TILE + LBL), (255, 255, 255))
        row_img.paste(tile(src, idn), (0, 0))
        for j, b in enumerate(backends):
            sw = cv2.imread(f"{OUT}/swaps/{b}/{idn}.png")
            r = cell.get((b, idn))
            if sw is None or r is None:
                continue
            lbl = f"{b[:13]} {r['mode'][:4]} {r['id_cos']}"
            row_img.paste(tile(sw, lbl), (TILE * (j + 1), 0))
        out_rows.append(row_img)
    H = sum(r.height for r in out_rows)
    coll = Image.new("RGB", (TILE * ncol, H), (255, 255, 255))
    y = 0
    for r in out_rows:
        coll.paste(r, (0, y))
        y += r.height
    coll.save(f"{OUT}/collage.png")
    print(f"wrote {OUT}/collage.png", flush=True)


if __name__ == "__main__":
    main()
