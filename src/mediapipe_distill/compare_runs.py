"""Side-by-side comparison of two validate_as_loss reports.

Renders a per-channel + per-atom table showing R² for both runs and the
ship-tier transition (ship / do_not_ship / degenerate). Designed for
v1 ↔ v2c, but works on any two reports of the same shape.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SHIP_TIERS = ("confident_ship", "ship", "do_not_ship", "degenerate")


def channel_to_tier(report: dict, channel: str) -> tuple[str, float]:
    s = report["layer_1_1_shippability"]
    for tier in SHIP_TIERS:
        for entry in s.get(tier, []):
            if entry["channel"] == channel:
                return tier, float(entry["r2"])
    return "unknown", float("nan")


def atom_to_tier(report: dict, tag: str) -> tuple[str, float]:
    a = report.get("layer_1_1_atom_shippability", {})
    if not a.get("available"):
        return "n/a", float("nan")
    for tier in SHIP_TIERS:
        for entry in a.get(tier, []):
            if entry["tag"] == tag:
                return tier, float(entry["r2"])
    return "unknown", float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, required=True, help="report A (e.g. v1)")
    p.add_argument("--b", type=Path, required=True, help="report B (e.g. v2c)")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    A = json.load(args.a.open())
    B = json.load(args.b.open())

    sa = A["layer_1_1_shippability"]
    all_channels = sorted(set(
        e["channel"]
        for tier in SHIP_TIERS
        for e in sa.get(tier, [])
    ))

    lines = []
    lines.append(f"# {args.label_a} vs {args.label_b}\n")
    lines.append(f"- {args.label_a}: `{args.a}`")
    lines.append(f"- {args.label_b}: `{args.b}`\n")

    aA = A["layer_1_1_atom_shippability"]
    aB = B["layer_1_1_atom_shippability"]
    if aA.get("available") and aB.get("available"):
        lines.append(f"## Atom shippability (8 NMF atoms)\n")
        lines.append(f"| atom_idx | tag | {args.label_a} R² | tier | "
                     f"{args.label_b} R² | tier | Δ |")
        lines.append("|---:|---|---:|---|---:|---|---:|")
        for tag in aA["atom_tags"]:
            ta, ra = atom_to_tier(A, tag)
            tb, rb = atom_to_tier(B, tag)
            d = (rb - ra) if not (ra != ra or rb != rb) else float("nan")
            lines.append(f"| atom | {tag} | {ra:.3f} | {ta} | {rb:.3f} | {tb} | {d:+.3f} |")
        lines.append("")

    lines.append(f"## Channel shippability (52 channels)\n")
    lines.append(f"| channel | {args.label_a} R² | tier | "
                 f"{args.label_b} R² | tier | Δ | move |")
    lines.append("|---|---:|---|---:|---|---:|---|")
    moves = {
        "promoted": 0, "demoted": 0, "stable_ship": 0,
        "stable_no_ship": 0, "stable_degenerate": 0,
    }
    rank = {"degenerate": 0, "do_not_ship": 1, "ship": 2, "confident_ship": 3}
    for ch in all_channels:
        ta, ra = channel_to_tier(A, ch)
        tb, rb = channel_to_tier(B, ch)
        d = (rb - ra) if not (ra != ra or rb != rb) else float("nan")
        ra_n = rank.get(ta, -1)
        rb_n = rank.get(tb, -1)
        if rb_n > ra_n:
            move = "↑ promoted"
            moves["promoted"] += 1
        elif rb_n < ra_n:
            move = "↓ demoted"
            moves["demoted"] += 1
        elif ta in ("confident_ship", "ship"):
            move = "= stable ship"
            moves["stable_ship"] += 1
        elif ta == "degenerate":
            move = "= degenerate"
            moves["stable_degenerate"] += 1
        else:
            move = "= no-ship"
            moves["stable_no_ship"] += 1
        lines.append(f"| {ch} | {ra:.3f} | {ta} | {rb:.3f} | {tb} | "
                     f"{d:+.3f} | {move} |")
    lines.append("")

    lines.append("## Summary")
    lines.append(f"- channels promoted: **{moves['promoted']}**")
    lines.append(f"- channels demoted: **{moves['demoted']}**")
    lines.append(f"- stable ship: {moves['stable_ship']}")
    lines.append(f"- stable no-ship: {moves['stable_no_ship']}")
    lines.append(f"- stable degenerate: {moves['stable_degenerate']}")
    lines.append("")
    lines.append(f"### Aggregate")
    keys = ["val_mse", "r2_median", "r2_mean", "frac_above_0p5", "frac_above_0p7"]
    lines.append(f"| metric | {args.label_a} | {args.label_b} | Δ |")
    lines.append("|---|---:|---:|---:|")
    for k in keys:
        a = A["layer_1_2_aggregate"].get(k, float("nan"))
        b = B["layer_1_2_aggregate"].get(k, float("nan"))
        lines.append(f"| {k} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(text)
        print(f"\nwrote {args.out_md}")


if __name__ == "__main__":
    main()
