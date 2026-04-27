"""CLIP zero-shot probe scorer for drift-attribute measurement.

For each PNG under `output/demographic_pc/overnight_drift/`, encode the
image once with OpenCLIP and compute a paired-prompt logit margin for a
bank of attributes. Writes:

    output/demographic_pc/clip_probes.parquet
        columns: rel_from_overnight, <probe>_margin for each probe

`<probe>_margin` = logit(positive_prompt) - logit(negative_prompt). Higher
margin = attribute more present. No training; probes are text pairs.

Probe bank covers the drift-framework attributes from
`docs/research/2026-04-23-drift-framework.md`:

    bearded, clean_shaven, long_hair, wrinkled, glasses,
    open_mouth, eyes_closed, smiling

Usage:
    uv run python -m src.demographic_pc.score_clip_probes --smoke      # 6 images
    uv run python -m src.demographic_pc.score_clip_probes --run        # full tree
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OVERNIGHT = ROOT / "output/demographic_pc/overnight_drift"
OUT_PARQUET_OPENCLIP = ROOT / "output/demographic_pc/clip_probes_openclip.parquet"
OUT_PARQUET_SIGLIP2  = ROOT / "output/demographic_pc/clip_probes_siglip2.parquet"

OPENCLIP_MODEL = "ViT-L-14"
OPENCLIP_PRETRAINED = "openai"
SIGLIP2_MODEL = "google/siglip2-so400m-patch16-384"

# (probe_name, positive_prompt, negative_prompt)
PROBES: list[tuple[str, str, str]] = [
    ("bearded",
     "a photo of a man with a thick full beard",
     "a photo of a clean-shaven man with no facial hair"),
    ("heavy_beard",
     "a photo of a man with a heavy bushy beard covering the jaw",
     "a photo of a man with smooth skin and no beard"),
    ("mustache_only",
     "a photo of a man with only a mustache",
     "a photo of a man with a full beard"),
    ("smiling",
     "a photo of a person smiling with teeth visible",
     "a photo of a person with a neutral closed mouth"),
    ("open_mouth",
     "a photo of a person with their mouth open wide",
     "a photo of a person with their mouth closed"),
    ("eyes_closed",
     "a photo of a person with their eyes closed",
     "a photo of a person with their eyes open"),
    ("glasses",
     "a photo of a person wearing eyeglasses",
     "a photo of a person not wearing glasses"),
    ("wrinkled",
     "a photo of an elderly person with deep wrinkles",
     "a photo of a young person with smooth skin"),
    ("long_hair",
     "a photo of a person with long hair past the shoulders",
     "a photo of a person with short cropped hair"),
    ("angry",
     "a photo of a person with an angry furious expression",
     "a photo of a person with a calm neutral expression"),
    ("surprised",
     "a photo of a person with a surprised shocked expression",
     "a photo of a person with a calm neutral expression"),
    ("puckered_lips",
     "a photo of a person with lips puckered forward",
     "a photo of a person with relaxed lips"),
]


class OpenClipBackend:
    name = "openclip-vitl14"

    def __init__(self, device: str):
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAINED, device=device)
        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)
        self.device = device

    @torch.no_grad()
    def encode_probes(self):
        out = {}
        for name, pos, neg in PROBES:
            tokens = self.tokenizer([pos, neg]).to(self.device)
            feat = self.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            out[name] = feat
        return out

    @torch.no_grad()
    def encode_image(self, path: Path):
        img = Image.open(path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        return feat / feat.norm(dim=-1, keepdim=True)


class Siglip2Backend:
    name = "siglip2-so400m"

    def __init__(self, device: str):
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(SIGLIP2_MODEL).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL)
        self.device = device

    @staticmethod
    def _as_tensor(x):
        # transformers >=5 returns BaseModelOutputWithPooling for some paths;
        # older returns a plain Tensor. Handle both.
        if isinstance(x, torch.Tensor):
            return x
        for attr in ("pooler_output", "last_hidden_state"):
            v = getattr(x, attr, None)
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError(f"cannot extract tensor from {type(x)}")

    @torch.no_grad()
    def encode_probes(self):
        out = {}
        for name, pos, neg in PROBES:
            inputs = self.processor(text=[pos, neg], return_tensors="pt",
                                    padding="max_length", truncation=True).to(self.device)
            feat = self._as_tensor(self.model.get_text_features(**inputs))
            feat = feat / feat.norm(dim=-1, keepdim=True)
            out[name] = feat
        return out

    @torch.no_grad()
    def encode_image(self, path: Path):
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        feat = self._as_tensor(self.model.get_image_features(**inputs))
        return feat / feat.norm(dim=-1, keepdim=True)


@torch.no_grad()
def score_image(backend, probe_feats, path: Path) -> dict[str, float]:
    feat = backend.encode_image(path)  # (1, D)
    out = {}
    for name, pf in probe_feats.items():
        sims = (feat @ pf.T).squeeze(0)
        out[f"{name}_margin"] = float(sims[0] - sims[1])
    return out


def iter_pngs(root: Path):
    for p in sorted(root.rglob("*.png")):
        if "collage" in p.name:
            continue
        yield p


SMOKE_PICKS = [
    ("bearded-should-be-high",
     OVERNIGHT / "beard" / "remove" / "elderly_latin_m" / "seed2026_s+0.00.png"),
    ("clean-shaven-should-be-low-beard",
     OVERNIGHT / "beard" / "add" / "asian_m" / "seed2026_s+0.00.png"),
    ("asian_m-with-beard-added-scale1",
     OVERNIGHT / "beard" / "add" / "asian_m" / "seed2026_s+1.00.png"),
    ("manic-smile-should-be-high-smile",
     OVERNIGHT / "smile" / "manic" / "young_european_f" / "seed2026_s+1.00.png"),
    ("neutral-should-be-low-smile",
     OVERNIGHT / "smile" / "faint" / "young_european_f" / "seed2026_s+0.00.png"),
    ("surprise-should-be-high-open-mouth",
     OVERNIGHT / "rebalance_reseed" / "surprise" / "black_f" / "seed2026_s+1.20.png"),
]


def smoke_test(backend, probe_feats) -> None:
    print(f"\n[{backend.name}] smoke")
    print(f"{'label':<48} {'bearded':>8} {'smiling':>8} {'open_mth':>8} {'surprsd':>8} {'wrinkld':>8}")
    for label, p in SMOKE_PICKS:
        if not p.exists():
            print(f"{label:<48}  MISSING {p}")
            continue
        r = score_image(backend, probe_feats, p)
        print(f"{label:<48} {r['bearded_margin']:>+8.3f} {r['smiling_margin']:>+8.3f} "
              f"{r['open_mouth_margin']:>+8.3f} {r['surprised_margin']:>+8.3f} "
              f"{r['wrinkled_margin']:>+8.3f}")


def run_full(backend, probe_feats, out_parquet: Path) -> None:
    pngs = list(iter_pngs(OVERNIGHT))
    print(f"[{backend.name}] scoring {len(pngs):,} images")
    rows = []
    t0 = time.time()
    for i, p in enumerate(pngs):
        r = score_image(backend, probe_feats, p)
        r["rel_from_overnight"] = str(p.relative_to(OVERNIGHT))
        rows.append(r)
        if (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(pngs)}] {(i+1)/dt:.1f} img/s  eta {(len(pngs)-i-1)/((i+1)/dt):.0f}s")
    df = pd.DataFrame(rows)
    cols = ["rel_from_overnight"] + [c for c in df.columns if c.endswith("_margin")]
    df = df[cols]
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False, compression="zstd")
    print(f"[save] → {out_parquet}  rows={len(df)}  cols={df.shape[1]}")


def get_backend(name: str, device: str):
    if name == "openclip":
        return OpenClipBackend(device), OUT_PARQUET_OPENCLIP
    if name == "siglip2":
        return Siglip2Backend(device), OUT_PARQUET_SIGLIP2
    raise ValueError(f"unknown backend {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backend", choices=["openclip", "siglip2", "both"],
                    default="siglip2")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backends = ["openclip", "siglip2"] if args.backend == "both" else [args.backend]
    for b in backends:
        print(f"\n[{b}] loading on {device}")
        backend, out_parquet = get_backend(b, device)
        probe_feats = backend.encode_probes()
        if args.smoke:
            smoke_test(backend, probe_feats)
        if args.run:
            run_full(backend, probe_feats, out_parquet)
    if not (args.smoke or args.run):
        ap.print_help()


if __name__ == "__main__":
    main()
