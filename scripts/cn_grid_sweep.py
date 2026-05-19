"""CN-strength x steps grid sweep across the importer identities.

One-shot Z-Image generation (txt2img + Z-Image Fun ControlNet driven by the
per-identity source canny), then inswapper identity swap + measurement.

Runs against a *remote* ComfyUI (the videocard Windows box) -- control images
are staged via the /upload/image API, not written to a local input dir.

Resumable: each (identity, strength, steps) cell appends one JSONL row;
on restart, completed cells are skipped.
"""
import io, json, os, sys, time, uuid
import cv2, numpy as np, requests
from PIL import Image, ImageDraw, ImageFont

os.chdir("/home/newub/w/vamp-interface")
sys.path.insert(0, "scripts")
from swap_core import (_crop_region, crop_and_upscale, detect_source,
                       load_swapper, make_face_app, mediapipe_kps_bbox,
                       swap_identity)

URL = os.environ.get("COMFY_URL", "http://192.168.87.25:8188")
OUT = "exp_output/cn_grid"
RENDERS = f"{OUT}/renders"
SWAPS = f"{OUT}/swaps"
RESULTS = f"{OUT}/results.jsonl"
os.makedirs(RENDERS, exist_ok=True)
os.makedirs(SWAPS, exist_ok=True)

PROMPT = ("a vibrant traditional Russian matryoshka nesting doll, glossy red "
          "and gold lacquer, ornate floral painting, with a realistic "
          "photographic human face, soft three-dimensional shading, correct "
          "facial proportions, defined nose and lips, natural-sized eyes, "
          "centered frontal face, wooden doll, plain background")

# Grid axes
STRENGTHS = [0.50, 0.70, 0.90]
STEPS = [6, 8]
IDENTITIES = [f"id_{i:02d}" for i in range(20)]

INSWAPPER = os.path.expanduser("~/w/ComfyUI/models/insightface/inswapper_128.onnx")
REF_DOLL = ("exp_output/matryoshka_bakeoff/renders/"
            "zimage_turbo_st06_euler_simple_seed74029470.png")


def control_canny(src_bgr, src_face, rect, canvas_hw):
    """Source-identity face canny placed in `rect` on a black canvas."""
    x0, y0, x1, y1 = rect
    sx0, sy0, sx1, sy1 = src_face.bbox.astype(int)
    m = int(0.35 * max(sx1 - sx0, sy1 - sy0))
    sh, sw = src_bgr.shape[:2]
    face = src_bgr[max(0, sy0 - m):min(sh, sy1 + m),
                   max(0, sx0 - m):min(sw, sx1 + m)]
    face = cv2.resize(face, (x1 - x0, y1 - y0))
    edges = cv2.Canny(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), 80, 160)
    canvas = np.zeros((*canvas_hw, 3), np.uint8)
    canvas[y0:y1, x0:x1] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return canvas


def upload_image(bgr, name):
    """Stage a control image into the remote ComfyUI input dir via API."""
    ok, buf = cv2.imencode(".png", bgr)
    r = requests.post(f"{URL}/upload/image",
                      files={"image": (name, io.BytesIO(buf.tobytes()),
                                       "image/png")},
                      data={"overwrite": "true"}, timeout=30)
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def build_wf(ctrl_name, *, seed, strength, steps, prefix):
    wf = json.load(open("comfyui/workflows/matryoshka_zimage_turbo.api.json"))
    subs = {"$$POSITIVE_PROMPT": PROMPT, "$$SEED": seed, "$$STEPS": steps,
            "$$SAMPLER": "euler", "$$SCHEDULER": "simple",
            "$$OUTPUT_PREFIX": prefix}
    for n in wf.values():
        for k, v in n.get("inputs", {}).items():
            if isinstance(v, str) and v in subs:
                n["inputs"][k] = subs[v]
    ks = wf["8"]
    model_src = ks["inputs"]["model"]
    wf["20"] = {"class_type": "ModelPatchLoader",
                "inputs": {"name": "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"}}
    wf["21"] = {"class_type": "LoadImage", "inputs": {"image": ctrl_name}}
    wf["22"] = {"class_type": "ZImageFunControlnet",
                "inputs": {"model": model_src, "model_patch": ["20", 0],
                           "vae": ["3", 0], "strength": strength,
                           "image": ["21", 0]}}
    ks["inputs"]["model"] = ["22", 0]
    return wf


def run(wf):
    cid = str(uuid.uuid4())
    r = requests.post(f"{URL}/prompt", json={"prompt": wf, "client_id": cid},
                      timeout=30)
    if r.status_code != 200:
        return None, r.text[:300]
    pid = r.json()["prompt_id"]
    for _ in range(600):
        h = requests.get(f"{URL}/history/{pid}", timeout=10).json()
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") != "success":
                return None, [m for m in st.get("messages", [])
                              if m[0] == "execution_error"]
            for o in h[pid].get("outputs", {}).values():
                for im in o.get("images", []):
                    return im, None
            return None, "no image"
        time.sleep(1)
    return None, "timeout"


def fetch(im):
    r = requests.get(f"{URL}/view", params={
        "filename": im["filename"], "subfolder": im.get("subfolder", ""),
        "type": "output"}, timeout=60)
    r.raise_for_status()
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


def id_cos(app, bgr, src_emb):
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
        if not line:
            continue
        d = json.loads(line)
        keys.add((d["identity"], d["strength"], d["steps"]))
    return keys


def main():
    app = make_face_app()
    swapper = load_swapper(INSWAPPER)

    ref = cv2.imread(REF_DOLL)
    _kps, _bbox = mediapipe_kps_bbox(ref)
    rect = _crop_region(_bbox, ref.shape, margin_frac=0.35)
    canvas_hw = ref.shape[:2]

    done = done_keys()
    cells = [(idn, s, st) for idn in IDENTITIES
             for st in STEPS for s in STRENGTHS]
    todo = [c for c in cells if c not in done]
    print(f"grid: {len(cells)} cells, {len(done)} done, {len(todo)} to run",
          flush=True)

    t_start = time.time()
    for i, (idn, strength, steps) in enumerate(todo):
        idx = int(idn.split("_")[1])
        seed = 91_000_000 + idx
        src_bgr = cv2.imread(f"data/importer/identities/{idn}.png")
        src_face = detect_source(app, src_bgr)
        src_emb = src_face.normed_embedding

        canny = control_canny(src_bgr, src_face, rect, canvas_hw)
        ctrl_name = upload_image(canny, f"cn_grid_{idn}.png")

        tag = f"{idn}_str{int(strength*100)}_st{steps}"
        t0 = time.time()
        wf = build_wf(ctrl_name, seed=seed, strength=strength, steps=steps,
                      prefix=f"cn_grid/{tag}")
        im, err = run(wf)
        if im is None:
            print(f"  [FAIL] {tag}: {err}", flush=True)
            continue
        doll = fetch(im)
        render_s = time.time() - t0
        cv2.imwrite(f"{RENDERS}/{tag}.png", doll)

        res, mode, det = swap_identity(app, swapper, doll, src_face)
        cv2.imwrite(f"{SWAPS}/{tag}.png", res)
        cos = id_cos(app, res, src_emb)

        row = {"identity": idn, "strength": strength, "steps": steps,
               "seed": seed, "mode": mode, "det": round(float(det), 4),
               "id_cos": round(float(cos), 4) if cos == cos else None,
               "render_s": round(render_s, 1)}
        with open(RESULTS, "a") as f:
            f.write(json.dumps(row) + "\n")
        eta = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1)
        print(f"  [{i+1}/{len(todo)}] {tag}  {mode:8s} det={det:.3f} "
              f"cos={cos:.3f} render={render_s:.1f}s  eta={eta/60:.1f}min",
              flush=True)

    print(f"done: {len(todo)} cells in {(time.time()-t_start)/60:.1f}min",
          flush=True)
    build_collage(app)


def build_collage(app):
    """Per-identity row: source + swap thumbnail per cell."""
    rows = json.loads("[" + ",".join(
        l for l in open(RESULTS).read().splitlines() if l.strip()) + "]")
    by_id = {}
    for r in rows:
        by_id.setdefault(r["identity"], []).append(r)
    cells_order = [(s, st) for st in STEPS for s in STRENGTHS]
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

    ncol = 1 + len(cells_order)
    out_rows = []
    for idn in sorted(by_id):
        src = cv2.imread(f"data/importer/identities/{idn}.png")
        row_img = Image.new("RGB", (TILE * ncol, TILE + LBL), (255, 255, 255))
        row_img.paste(tile(src, idn), (0, 0))
        cellmap = {(r["strength"], r["steps"]): r for r in by_id[idn]}
        for j, key in enumerate(cells_order):
            r = cellmap.get(key)
            if r is None:
                continue
            tag = (f"{idn}_str{int(key[0]*100)}_st{key[1]}")
            sw = cv2.imread(f"{SWAPS}/{tag}.png")
            if sw is None:
                continue
            cos = r["id_cos"]
            lbl = f"s{key[0]} st{key[1]} {r['mode'][:4]} {cos}"
            row_img.paste(tile(sw, lbl), (TILE * (j + 1), 0))
        out_rows.append(row_img)
    H = sum(r.height for r in out_rows)
    coll = Image.new("RGB", (TILE * ncol, H), (255, 255, 255))
    y = 0
    for r in out_rows:
        coll.paste(r, (0, y))
        y += r.height
    coll.save(f"{OUT}/grid_collage.png")
    print(f"wrote {OUT}/grid_collage.png", flush=True)


if __name__ == "__main__":
    main()
