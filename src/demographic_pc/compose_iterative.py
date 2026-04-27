"""Iterative corrective composition (v0).

Runs the threshold-gated residual-descent loop described in
`docs/research/2026-04-23-framework-procedure.md` ("The algorithm").

Given:
  - a base (maps to an EVAL_BASES prompt) + seed
  - a primary prompt-pair + target scale
  - one or more candidate counter-pairs, each with a readout name,
    violation threshold, and scale-bump schedule

The loop:
  t=0   render primary-only, score.
  t=t+1 for each confound whose |readout| > threshold:
          if not yet active → activate counter at scale_init
          if already active and still violating → bump scale by scale_step (cap at scale_max)
          if already active and now under threshold → lock (stop bumping)
        render multi-pair with primary + all active counters, re-score.
        stop when no confound violates, or t == max_iters.

Writes per-iteration PNG + pkl + row in a scores parquet, plus a
markdown report summarising what each iteration changed.

Spec YAML:
  name: smile_on_young_f_with_age_hold
  base: young_european_f
  seed: 2026
  max_iters: 4
  primary:
    label: smile_v_eur_me
    pos: "An {age} {ethnicity} {gender} smiling warmly."
    neg: "An {age} Middle Eastern {gender} smiling warmly."
    scale: 0.5
  confounds:
    - name: age_drift
      readout: ins_age          # any column in the scored dataframe
      threshold: 4.0            # |ins_age - anchor_ins_age| > 4.0 y
      diff_from_anchor: true    # compare to scale=0 anchor (default false = raw magnitude)
      counter:
        label: age_hold_adult
        pos: "An adult European {gender}, neutral expression."
        neg: "A young European {gender}, neutral expression."
      scale_init: 0.25
      scale_step: 0.15
      scale_max: 0.75

Usage:
  uv run python -m src.demographic_pc.compose_iterative --spec specs/compose_smile_young_f.yaml
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml

from src.demographic_pc.promptpair_iterate import (
    EVAL_BASES, _load_scorers, BASE_AGE_WORDS, BASE_ETHNICITY_WORDS, BASE_GENDER_WORDS,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "output/demographic_pc/compose_iterative"


def _resolve(prompt: str, base_name: str) -> str:
    return prompt.format(
        age=BASE_AGE_WORDS.get(base_name, "adult"),
        ethnicity=BASE_ETHNICITY_WORDS.get(base_name, ""),
        gender=BASE_GENDER_WORDS.get(base_name, "person"),
    )


def _score_one(png_path: Path, anchor_png: Path | None, scorers) -> dict:
    """Score a single image, computing identity_cos vs the anchor if provided.
    Returns a flat dict of readouts."""
    import cv2
    import numpy as np
    from src.demographic_pc.score_blendshapes import make_landmarker, score_png
    from src.demographic_pc.build_sample_index import project_sample

    (nmf, mv, ff, ins, sig_backend, probe_feats) = scorers
    (W, W_pinv, H_fit, h_lookup, mu, sigma, channels_raw, channels_full,
     prune_mask, base_idx) = nmf

    def _score(p: Path) -> dict:
        bgr = cv2.imread(str(p))
        row: dict = {"img_path": str(p)}
        with make_landmarker() as lm:
            bs = score_png(lm, p) if bgr is not None else None
        if bs is not None:
            for c in channels_full:
                row[f"bs_{c}"] = float(bs.get(c, 0.0))
        if bgr is not None:
            m = mv.predict(bgr);  row["mv_age"] = m["age"]
            f = ff.predict(bgr);  row["ff_race"] = f["race"]; row["ff_gender"] = f["gender"]
            i = ins.predict(bgr); row["ins_age"] = i["age"]; row["ins_gender"] = i["gender"]
            row["_ins_embedding"] = i["embedding"] if i["embedding"] is not None else None
            feat = sig_backend.encode_image(p)
            for name, pf in probe_feats.items():
                sims = (feat @ pf.T).squeeze(0)
                row[f"siglip_{name}_margin"] = float(sims[0] - sims[1])
            row["_siglip_feat"] = feat.squeeze(0).cpu().numpy().astype(np.float32)
        return row

    cur = _score(png_path)
    if anchor_png is not None and anchor_png.exists():
        anc = _score(anchor_png)
        for embed_key, out_key in (("_ins_embedding", "identity_cos_to_base"),
                                    ("_siglip_feat", "siglip_img_cos_to_base")):
            a = anc.get(embed_key); b = cur.get(embed_key)
            if a is not None and b is not None:
                a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
                na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
                if na * nb > 1e-9:
                    cur[out_key] = float((a * b).sum() / (na * nb))
        # anchor-relative numeric deltas for readouts the spec compares via diff_from_anchor
        for k in ("mv_age", "ins_age"):
            if k in cur and k in anc:
                cur[f"{k}__anchor"] = anc[k]
                cur[f"{k}__delta"] = cur[k] - anc[k]
    for k in ("_ins_embedding", "_siglip_feat"):
        cur.pop(k, None)
    return cur


def _violating(row: dict, readout: str, threshold, diff_from_anchor: bool) -> tuple[bool, float]:
    """Threshold can be a scalar (absolute-value check) or a dict of
    comparison ops: {">": x}, {"<": x}, {">=": x}, {"<=": x}, {"abs>": x}.
    Multiple ops in the dict are OR'd."""
    key = f"{readout}__delta" if diff_from_anchor else readout
    val = row.get(key)
    if val is None:
        return False, float("nan")
    try:
        v = float(val)
    except (TypeError, ValueError):
        return False, float("nan")
    if isinstance(threshold, dict):
        violating = False
        for op, t in threshold.items():
            t = float(t)
            if op == ">"  and v >  t: violating = True
            elif op == ">=" and v >= t: violating = True
            elif op == "<"  and v <  t: violating = True
            elif op == "<=" and v <= t: violating = True
            elif op == "abs>" and abs(v) > t: violating = True
        return violating, v
    return abs(v) > float(threshold), v


async def run(spec_path: Path) -> None:
    from src.demographic_pc.comfy_flux import ComfyClient
    from src.demographic_pc.fluxspace_metrics import (
        pair_measure_workflow, pair_multi_measure_workflow,
    )

    scorers = _load_scorers()

    spec = yaml.safe_load(spec_path.read_text())
    name = spec["name"]
    base_name = spec["base"]
    seed = int(spec["seed"])
    max_iters = int(spec.get("max_iters", 4))

    base_prompt = dict(EVAL_BASES).get(base_name)
    if base_prompt is None:
        raise SystemExit(f"base '{base_name}' not in EVAL_BASES")

    primary = spec["primary"]
    confounds = spec.get("confounds", [])

    run_dir = OUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    # active counter state: name → {scale, locked}
    active: dict[str, dict] = {}

    async with ComfyClient() as client:
        anchor_png = run_dir / f"anchor_seed{seed}.png"
        anchor_pkl = anchor_png.with_suffix(".pkl")
        if not anchor_png.exists():
            # scale=0 short-circuits FluxSpaceEditPair → base-only render
            wf = pair_measure_workflow(
                seed, str(anchor_pkl), f"compose_{name}_anchor",
                base_prompt=base_prompt, edit_a=base_prompt, edit_b=base_prompt,
                scale=0.0, start_percent=0.15, end_percent=1.0)
            await client.generate(wf, anchor_png)

        primary_pos = _resolve(primary["pos"], base_name)
        primary_neg = _resolve(primary["neg"], base_name)
        primary_scale = float(primary["scale"])
        primary_label = primary.get("label", "primary")

        iterations: list[dict] = []
        for t in range(max_iters + 1):
            iter_png = run_dir / f"iter{t:02d}_seed{seed}.png"
            iter_pkl = iter_png.with_suffix(".pkl")
            if t == 0:
                pairs_for_log = [{"label": primary_label, "scale": primary_scale}]
                if not iter_png.exists():
                    wf = pair_measure_workflow(
                        seed, str(iter_pkl), f"compose_{name}_iter{t:02d}",
                        base_prompt=base_prompt,
                        edit_a=primary_pos, edit_b=primary_neg,
                        scale=primary_scale, start_percent=0.15, end_percent=1.0)
                    await client.generate(wf, iter_png)
            else:
                pairs = [{
                    "edit_a": primary_pos, "edit_b": primary_neg,
                    "scale": primary_scale, "label": primary_label,
                }]
                pairs_for_log = [{"label": primary_label, "scale": primary_scale}]
                for c in confounds:
                    st = active.get(c["name"])
                    if st is None:
                        continue
                    ca = _resolve(c["counter"]["pos"], base_name)
                    cb = _resolve(c["counter"]["neg"], base_name)
                    pairs.append({
                        "edit_a": ca, "edit_b": cb,
                        "scale": st["scale"],
                        "label": c["counter"].get("label", c["name"]),
                    })
                    pairs_for_log.append({"label": c["counter"].get("label", c["name"]),
                                          "scale": st["scale"]})
                if len(pairs) > 4:
                    raise SystemExit(f"compose_iterative: {len(pairs)} pairs exceeds node limit (4)")
                if not iter_png.exists():
                    wf = pair_multi_measure_workflow(
                        seed, str(iter_pkl), f"compose_{name}_iter{t:02d}",
                        base_prompt=base_prompt, pairs=pairs,
                        start_percent=0.15, end_percent=1.0)
                    await client.generate(wf, iter_png)

            print(f"[iter {t:02d}] scoring...")
            scores = _score_one(iter_png, anchor_png, scorers)

            iter_row = {"iter": t, "pairs": pairs_for_log}
            any_violating = False
            confound_detail = []
            for c in confounds:
                viol, val = _violating(scores, c["readout"], c["threshold"],
                                       bool(c.get("diff_from_anchor", False)))
                st = active.get(c["name"])
                action = ""
                if t == 0:
                    # on iter 0, activate any violating confound at scale_init
                    if viol:
                        active[c["name"]] = {"scale": float(c["scale_init"]), "locked": False}
                        action = f"activate @ s={c['scale_init']}"
                        any_violating = True
                else:
                    if viol:
                        if st is None:
                            # newly emerged — activate
                            active[c["name"]] = {"scale": float(c["scale_init"]), "locked": False}
                            action = f"activate @ s={c['scale_init']} (newly emerged)"
                            any_violating = True
                        elif st["locked"]:
                            action = "locked (was under, still under? bump skipped)"
                        else:
                            new_scale = min(st["scale"] + float(c["scale_step"]),
                                             float(c["scale_max"]))
                            if new_scale > st["scale"] + 1e-6:
                                st["scale"] = new_scale
                                action = f"bump → s={new_scale:.3f}"
                            else:
                                action = f"at cap s={new_scale:.3f}; no further bump"
                            any_violating = True
                    else:
                        if st is not None and not st["locked"]:
                            st["locked"] = True
                            action = f"lock @ s={st['scale']:.3f}"
                        elif st is not None:
                            action = f"locked @ s={st['scale']:.3f}"
                confound_detail.append({
                    "name": c["name"], "readout": c["readout"], "value": val,
                    "threshold": c["threshold"],
                    "violating": viol, "action": action,
                })
            iter_row["confounds"] = confound_detail
            iter_row["target_readout"] = scores.get(primary.get("target_readout", "siglip_smiling_margin"))
            iter_row["identity_cos_to_base"] = scores.get("identity_cos_to_base")
            iter_row["ff_race"] = scores.get("ff_race")
            iterations.append(iter_row)

            if t > 0 and not any_violating:
                print(f"[iter {t:02d}] all confounds below threshold — stopping")
                break

    _write_report(run_dir, spec, iterations, anchor_png)
    print(f"[save] → {run_dir}")


def _write_report(run_dir: Path, spec: dict, iterations: list[dict], anchor_png: Path) -> None:
    lines: list[str] = []
    lines.append("---\nstatus: live\ntopic: metrics-and-direction-quality\n---\n")
    lines.append(f"# Iterative composition — `{spec['name']}`\n")
    lines.append(f"- base: `{spec['base']}`  seed: {spec['seed']}  max_iters: {spec.get('max_iters', 4)}")
    lines.append(f"- primary: `{spec['primary'].get('label', 'primary')}` @ s={spec['primary']['scale']}")
    lines.append(f"- anchor: `{anchor_png.name}`")
    lines.append("")
    lines.append("## Iteration trace")
    lines.append("")
    lines.append("| iter | pairs (label@scale) | target | id_cos | race | confounds |")
    lines.append("|---|---|---|---|---|---|")
    for it in iterations:
        pairs_s = ", ".join(f"{p['label']}@{p['scale']:.2f}" for p in it["pairs"])
        tgt = it.get("target_readout")
        idc = it.get("identity_cos_to_base")
        race = it.get("ff_race", "?")
        conf_cells = []
        for c in it["confounds"]:
            mark = "⚠" if c["violating"] else "✓"
            conf_cells.append(f"{mark} {c['name']}={c['value']:.3f}→{c['action'] or 'hold'}")
        lines.append(f"| {it['iter']:02d} | {pairs_s} | "
                     f"{(f'{tgt:.3f}' if tgt is not None else '—')} | "
                     f"{(f'{idc:.3f}' if idc is not None else '—')} | "
                     f"{race} | " + "<br>".join(conf_cells) + " |")
    lines.append("")
    lines.append("## Renders")
    lines.append("")
    lines.append(f"- anchor: `{anchor_png.name}`")
    for it in iterations:
        lines.append(f"- iter {it['iter']:02d}: `iter{it['iter']:02d}_seed{spec['seed']}.png`")
    (run_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    args = ap.parse_args()
    asyncio.run(run(args.spec))


if __name__ == "__main__":
    main()
