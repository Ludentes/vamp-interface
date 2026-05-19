"""Stage-1 quick gate: is GHOST (sber-swap) good enough to bother with?

Runs the GHOST AEI_Net generator natively (vendored model code + the released
G_unet_2blocks / backbone weights) on a few CN-grid doll renders, scores
identity, and writes a collage. No GHOST env -- the generator and its iresnet100
ArcFace are plain torch state_dicts loaded under our torch 2.x.

Decision gate: proceed to a full swap_core bake-off only if GHOST is visually
clean and id_cos is in range (>= ~0.80). Otherwise GHOST is falsified.

Recipe ported from ghost/utils/inference/{core,image_processing,faceshifter_run}.py:
  - crop: insightface arcface align to 224 (face_align.estimate_norm, mode None)
  - source id: RGB crop -> (x-0.5)/0.5 -> downscale x0.5 -> iresnet100 -> 512-d
  - swap:   target RGB 224 -> (x-0.5)/0.5, half -> G(target, src_emb)
  - output: (y*0.5+0.5)*255 -> BGR uint8 224

id_cos is measured on the raw G output crop (before paste-back) so it isolates
the swapper from any blending choice.
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

os.chdir("/home/newub/w/vamp-interface")
sys.path.insert(0, "scripts")
sys.path.insert(0, "/home/newub/w/ghost")

from insightface.app.common import Face

from swap_core import make_face_app

from network.AEI_Net import AEI_Net
from arcface_model.iresnet import iresnet100

GHOST = "/home/newub/w/ghost"
OUT = "exp_output/ghost_gate"
DOLL = "exp_output/cn_grid/renders/{idn}_str90_st6.png"
SRC = "data/importer/identities/{idn}.png"
IDENTITIES = [f"id_{i:02d}" for i in range(5)]

# standard 5-point arcface template (112px), scaled to GHOST's 224 crop
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                        [56.0252, 71.7366], [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float32) * (224.0 / 112.0)


def load_ghost():
    G = AEI_Net("unet", num_blocks=2, c_id=512)
    G.eval()
    G.load_state_dict(torch.load(f"{GHOST}/weights/G_unet_2blocks.pth",
                                 map_location="cpu"))
    G = G.cuda().half()
    arc = iresnet100(fp16=False)
    arc.load_state_dict(torch.load(f"{GHOST}/arcface_model/backbone.pth",
                                   map_location="cpu"))
    arc = arc.cuda().eval()
    return G, arc


def crop224(app, bgr):
    """Arcface-align the largest detected face to a 224x224 BGR crop + M."""
    faces = app.get(bgr)
    if not faces:
        return None, None
    kps = max(faces, key=lambda f: f.det_score).kps
    M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)
    crop = cv2.warpAffine(bgr, M, (224, 224), borderValue=0.0)
    return crop, M


def src_embed(arc, crop_bgr):
    """GHOST source identity: RGB 224 crop -> (x-.5)/.5 -> x0.5 -> iresnet100."""
    rgb = crop_bgr[:, :, ::-1]
    t = torch.tensor(rgb.copy(), dtype=torch.float32).cuda() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    t = (t - 0.5) / 0.5
    t = F.interpolate(t, scale_factor=0.5, mode="bilinear", align_corners=True)
    with torch.no_grad():
        return arc(t)


def ghost_swap(G, emb, target_crop_bgr):
    """Run AEI_Net; returns a 224x224 BGR uint8 swapped face.

    The AEI_Net U-Net needs even downsampling, so GHOST's resize_frames feeds
    the generator at 256; get_final_image then resizes the output back to 224.
    """
    rgb = cv2.resize(target_crop_bgr, (256, 256))[:, :, ::-1]
    t = torch.from_numpy(rgb.copy()).cuda() / 255.0
    t = (t - 0.5) / 0.5
    t = t.permute(2, 0, 1).unsqueeze(0).half()
    with torch.no_grad():
        y, _ = G(t, emb.half())
        y = (y.permute(0, 2, 3, 1) * 0.5 + 0.5) * 255
        y = y[:, :, :, [2, 1, 0]].clamp(0, 255).type(torch.uint8)
    return cv2.resize(y[0].cpu().numpy(), (224, 224))


def id_cos(app, bgr, src_emb_if):
    """Post-swap insightface ArcFace cosine vs source (matches bake-off)."""
    faces = app.get(bgr)
    if faces:
        f = max(faces, key=lambda x: x.det_score)
        return float(np.dot(f.normed_embedding, src_emb_if))
    rec = app.models.get("recognition")
    if rec is None:
        return float("nan")
    h, w = bgr.shape[:2]
    kps = np.array([[w * .35, h * .4], [w * .65, h * .4], [w * .5, h * .55],
                    [w * .4, h * .7], [w * .6, h * .7]], dtype=np.float32)
    f = Face(bbox=np.array([0, 0, w, h], dtype=np.float32), kps=kps,
             det_score=1.0)
    rec.get(bgr, f)
    return float(np.dot(f.normed_embedding, src_emb_if))


def paste_back(doll, swap224, M):
    """Inverse-warp the 224 swap into the doll with a feathered oval mask."""
    h, w = doll.shape[:2]
    mask = np.zeros((224, 224), np.float32)
    cv2.ellipse(mask, (112, 112), (96, 116), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 12)
    Mi = cv2.invertAffineTransform(M)
    sw = cv2.warpAffine(swap224, Mi, (w, h), borderMode=cv2.BORDER_REPLICATE)
    mk = cv2.warpAffine(mask, Mi, (w, h))[:, :, None]
    return (mk * sw + (1 - mk) * doll).astype(np.uint8)


def main():
    os.makedirs(f"{OUT}/swaps", exist_ok=True)
    app = make_face_app()
    G, arc = load_ghost()

    rows = []
    print(f"{'identity':10s} {'id_cos':>8s}  notes")
    for idn in IDENTITIES:
        doll = cv2.imread(DOLL.format(idn=idn))
        src = cv2.imread(SRC.format(idn=idn))
        if doll is None or src is None:
            print(f"{idn:10s}  missing input"); continue

        src_crop, _ = crop224(app, src)
        tgt_crop, M = crop224(app, doll)
        if src_crop is None or tgt_crop is None:
            print(f"{idn:10s}  no face (src={src_crop is not None} "
                  f"tgt={tgt_crop is not None})"); continue

        # source identity for the insightface metric
        sf = max(app.get(src), key=lambda f: f.det_score)
        emb = src_embed(arc, src_crop)
        swap = ghost_swap(G, emb, tgt_crop)
        cos = id_cos(app, swap, sf.normed_embedding)
        pasted = paste_back(doll, swap, M)
        cv2.imwrite(f"{OUT}/swaps/{idn}_crop.png", swap)
        cv2.imwrite(f"{OUT}/swaps/{idn}_pasted.png", pasted)
        rows.append((idn, cos, src, doll, swap, pasted))
        print(f"{idn:10s} {cos:8.3f}")

    if rows:
        cosv = [c for _, c, *_ in rows]
        print(f"\nmean id_cos = {np.mean(cosv):.3f}  "
              f"min {np.min(cosv):.3f}  max {np.max(cosv):.3f}  (n={len(rows)})")
        build_collage(rows)


def build_collage(rows):
    from PIL import Image
    T = 224
    cols = 4  # src, doll, ghost-crop, ghost-pasted
    coll = Image.new("RGB", (T * cols, T * len(rows)), "white")
    for r, (idn, cos, src, doll, swap, pasted) in enumerate(rows):
        for c, im in enumerate([src, doll, swap, pasted]):
            t = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            t.thumbnail((T, T))
            coll.paste(t, (c * T + (T - t.width) // 2,
                           r * T + (T - t.height) // 2))
    coll.save(f"{OUT}/collage.png")
    print(f"wrote {OUT}/collage.png")


if __name__ == "__main__":
    main()
