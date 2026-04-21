"""Build the Stage 4.5 blog cover: one portrait, λ sweep, Ours (top) vs FluxSpace (bottom).

Columns: λ ∈ {−3, −2, −1, 0, +1, +2, +3}. Rows: Ours, FluxSpace.
λ=0 column is the shared baseline, shown in both rows.
Saved at docs/blog/images/2026-04-20-demographic-pc-extraction-end-to-end-cover.png,
overwriting the Stage-1-era placeholder.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RENDERS = ROOT / "output" / "demographic_pc" / "stage4_5" / "renders"
OUT = ROOT / "docs" / "blog" / "images" / "2026-04-20-demographic-pc-extraction-end-to-end-cover.png"

PORTRAIT = "p03_0_4_s2003"
LAMS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

CELL_W, CELL_H = 256, 342
LABEL_H = 32
PAD = 4
ROWS = 2

METHODS = [("Ours (ridge, manifold-aligned)", "ours"),
           ("FluxSpace-coarse (prompt-pair)", "fluxspace")]


def path_for(method: str, lam: float) -> Path:
    if lam == 0.0:
        return RENDERS / "baseline" / f"{PORTRAIT}__lam+0.00.png"
    return RENDERS / method / f"{PORTRAIT}__lam{lam:+.2f}.png"


def try_font(size: int) -> ImageFont.ImageFont:
    for cand in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> None:
    cols = len(LAMS)
    W = cols * CELL_W + (cols + 1) * PAD + 160   # leftmost label column
    H = LABEL_H + ROWS * (CELL_H + PAD + LABEL_H)
    canvas = Image.new("RGB", (W, H), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    font_col = try_font(18)
    font_row = try_font(18)

    # Column headers: λ values
    for j, lam in enumerate(LAMS):
        x = 160 + PAD + j * (CELL_W + PAD)
        label = f"λ = 0 (baseline)" if lam == 0.0 else f"λ = {lam:+.0f}"
        draw.text((x + 6, 6), label, fill=(210, 210, 215), font=font_col)

    for i, (label, method) in enumerate(METHODS):
        y0 = LABEL_H + i * (CELL_H + PAD + LABEL_H)
        # Row label (rotated or just left-aligned)
        draw.text((8, y0 + CELL_H // 2 - 10), label.split(" (")[0], fill=(230, 230, 235), font=font_row)
        draw.text((8, y0 + CELL_H // 2 + 10), "(" + label.split(" (")[1], fill=(160, 160, 165), font=font_row)
        for j, lam in enumerate(LAMS):
            p = path_for(method, lam)
            if not p.exists():
                print(f"MISSING: {p}")
                continue
            img = Image.open(p).convert("RGB").resize((CELL_W, CELL_H), Image.LANCZOS)
            x = 160 + PAD + j * (CELL_W + PAD)
            canvas.paste(img, (x, y0))

    # Suptitle strip at the very top? Keep it minimal.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, optimize=True)
    print(f"wrote {OUT}  size={canvas.size}")


if __name__ == "__main__":
    main()
