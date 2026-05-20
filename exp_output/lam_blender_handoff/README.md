# LAM Gaussian-Splat Repaint — Blender + Claude Experiment Kit

You're holding the canonical 3D Gaussian Splats from a head-avatar model called **LAM** (Large Avatar Model, SIGGRAPH 2025). Two human avatars, both produced from a single PNG anchor each by a learned encoder. The pipeline we're running animates these splats per-frame from an iPhone face-tracking signal — driving live photoreal-ish video.

## Watch this first

Open these two `.mp4`s before doing anything else:

- `renders/take2__asian_m__arkit600.mp4`
- `renders/take2__young_european_f__arkit600.mp4`

Each is a 20-second clip of an iPhone-driven face animating the corresponding canonical splat. These are your **"before"** — the unedited baseline. They look fine-ish and very "early-2010s game character." That plastic quality is exactly why we're doing this — real-people anchors don't impress; the real opportunity is stylized characters where uncanny valley isn't a target.

## The loop

```
   You edit  ─────►  edited.ply ─────►  Us
       ▲                                   │
       │                                   │ re-render against
       │                                   │ same 20s take
       │                                   ▼
   You watch  ◄─────────────── after.mp4 ─┘
```

You ship us an edited `.ply` + a one-line description + (ideally) a Blender viewport screenshot. Within an hour or so, we ship back an `after.mp4` showing your edit *moving across the same 20s of facial animation* as the "before" clip. You compare side-by-side, decide what to try next, iterate.

**The thing we want you to try:** take one of these `.ply` files into Blender (or wherever), and **stylize it**. Repaint it, retint hair, project an anime portrait onto it, prune it, swap colour palettes — whatever you can pull off. Then send the edited `.ply` back. We'll plug it back into the same pipeline and see how the edits hold up across full facial motion.

Why this is interesting: the LAM model only animates **xyz positions** per frame — opacity, rotation, scaling, spherical harmonics, and per-Gaussian offsets are all **shared with the canonical**. So if you change the colours on the canonical splat, every animated frame inherits your edit. The canonical `.ply` is literally an editable rest pose for an entire avatar's worth of video.

The current baseline output is "early-2010s game character" plastic. Real-people anchors don't really impress; the model's real opportunity is **stylized characters** where the uncanny valley isn't a target. So this is the start of the stylized-renderer thread on top of LAM.

---

## What's in this package

```
lam_blender_handoff/
├── README.md                                  ← this file
├── splats/
│   ├── asian_m_gs_offset.ply                  ← canonical splat, anchor 1
│   ├── asian_m_textured_mesh.obj              ← FLAME mesh w/ vertex colours
│   ├── asian_m_shaped_mesh.obj                ← bare FLAME geometry
│   ├── young_european_f_gs_offset.ply         ← canonical splat, anchor 2
│   ├── young_european_f_textured_mesh.obj
│   └── young_european_f_shaped_mesh.obj
├── anchors/
│   ├── asian_m.png                            ← source PNG the encoder saw
│   └── young_european_f.png
├── renders/
│   ├── take2__asian_m__arkit600.mp4           ← BEFORE: 20s animated, anchor 1
│   └── take2__young_european_f__arkit600.mp4  ← BEFORE: 20s animated, anchor 2
└── context/
    ├── 2026-05-13-lam-arkit-spike-resolved.md      ← how the pipeline works
    └── 2026-05-13-claude-blender-and-supersplat.md ← tool landscape
```

The `*_gs_offset.ply` files are the canonical splats — the things you'll be editing. The other two `.obj` files are bonus context showing the FLAME mesh that lives "underneath" the splats; you can use them as guides for masking (e.g. "select all splats near these head vertices").

The two `.mp4`s are the **"before"** — what the unedited splats look like when our pipeline animates them against a real 20-second iPhone face take. After you ship us your edited `.ply`, we re-render *the same take* with *the same camera and motion* and send you the matching **"after"** clip. Side-by-side comparison is the whole point — you'll see exactly how your colour / paint / prune choices ride across talking and head motion.

---

## What you can do, ranked roughly by interestingness

1. **Project a 2D image onto the splats.** Find a stylized portrait (anime, painting, illustration) that vaguely matches the head pose in the anchor PNG, and paint it onto the splat surface. KIRI's 3DGS Render addon for Blender has an image-based brush that does exactly this. The goal is a splat that, when animated, *looks like the stylized portrait moving*.
2. **Repaint by region.** Use the brush to retint hair, eyes, skin, mouth. Treat it like Photoshop on a 3D point cloud. Pick a stylized look and steer toward it.
3. **Prune.** There may be floater splats around the head (especially behind/below). Box-select and delete them. Cleaner silhouette.
4. **Push the colour grade.** KIRI's panel has brightness / contrast / saturation / hue-shift / per-channel curves. Push hard and see what survives export.
5. **Stretch goal — geometry.** LAM's animation only varies `xyz`, but the canonical splats *also* have positions. If you translate / scale a chunk (e.g. enlarge eyes for anime proportions), the relative offsets will carry into animation. Won't be perfect because the FLAME mesh underneath doesn't know about your edit, but it's a wild direction.

## What you ship back, and what you get

**You ship:**
- One or more **edited `_gs_offset.ply`** files, named distinctly (e.g. `young_european_f_anime_v1.ply`, `asian_m_hair_teal.ply`)
- A **one-line description** per file ("retinted hair to teal", "projected an anime portrait onto the face", "extreme saturation + pruned floaters")
- A **Blender viewport screenshot** per file showing the edit at rest

**You get back, per file:**
- An **`after.mp4`** — same 20-second take, same camera, same iPhone-driven motion, but rendered against your edited splat. Direct visual comparison against the "before" clip you already have.

Turnaround is roughly under an hour per delivery once you ship us a batch. Throw multiple variants at us at once — the per-render cost is small.

---

## Setup — verbose

### Software you'll need

- **Blender 4.3 or newer** (4.5 recommended). Free from blender.org.
- **KIRI 3DGS Render addon v4.1+** — `Kiri-Innovation/3dgs-render-blender-addon` on GitHub, or install via Blender Extensions. Free. This is the splat editor.
- **Claude Desktop** (the app, not the web UI), because we want the MCP server.
- **Python with `uv`** — needed by `blender-mcp`. If you don't have `uv`, install via `pipx install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **`ahujasid/blender-mcp`** — the community Blender ↔ Claude bridge. Configured below.

You can do the whole experiment in Blender by hand without Claude — KIRI's UI is fine. The Claude path is more interesting because you can iterate via chat.

### Step 1 — install Blender + KIRI addon

1. Download Blender 4.5 from `https://www.blender.org/download/` and install.
2. Launch Blender.
3. Open the KIRI addon. Easiest: `Edit → Preferences → Get Extensions`, search "3DGS Render", install. Alternatively: download the `.zip` from `https://github.com/Kiri-Innovation/3dgs-render-blender-addon/releases`, then `Edit → Preferences → Add-ons → Install from Disk → pick the zip → enable`.
4. After enabling, you'll see a **3DGS** tab in the right-hand `N` panel of the 3D viewport. If it isn't there, press `N` to toggle the panel.

### Step 2 — load a canonical splat

1. `File → Import → 3DGS Render (.ply)` (the KIRI addon adds this menu entry).
2. Pick `splats/young_european_f_gs_offset.ply`. When prompted, choose **"Import as Splats"** for full quality. (The "Import as Points" mode is faster but loses colour fidelity — useful for big crops, not for painting.)
3. Wait. The first import can take a minute. You'll get a head-shaped point cloud floating at the origin.
4. Optional but recommended: also `File → Import → Wavefront (.obj)` the matching `young_european_f_textured_mesh.obj`. This loads the underlying FLAME mesh as a regular Blender mesh. Hide it (`H`) when not needed — it's just a positional reference.

### Step 3 — basic navigation + cleanup

- Numpad `0` to look through the default camera. Middle-mouse-drag to orbit, `Shift+MMB` to pan, scroll to zoom.
- In the **3DGS panel** (right side, `N`), switch between **Edit Mode** (lets you select/delete splats) and **Render Mode** (real-time preview with full splat shader).
- To prune floaters: `Tab` into Edit Mode, then `B` for box-select or `C` for circle-select. Select unwanted splats, press `X` → Delete. Standard Blender mesh editing keys work here.

### Step 4 — colour edits

Two flavours of colour edit, both in the 3DGS panel:

- **Global grade.** Find the "Color Grading" sub-panel. Brightness / Contrast / Saturation / Hue / per-channel curves. Drag sliders, watch real-time update. Cheap and powerful.
- **Brush painting** (this is the interesting one). KIRI calls it the "3DGS Paint" workflow (v3.0+). You pick a brush colour or an image as the brush texture, then paint directly onto splats in the viewport. The colours stick to individual Gaussians and persist through export.

For image-based painting:
1. Find a stylized 2D portrait you want to project (e.g. an anime portrait facing roughly forward).
2. In the 3DGS panel, set the brush mode to "Image" and load your portrait as the brush texture.
3. Position the viewport so the splat head is facing the camera at roughly the same angle as your reference image.
4. Use **projection paint** (KIRI calls this out specifically): paint across the splat surface from the camera angle, and the brush samples colours from the image texture and writes them into the splats' base colour (DC spherical harmonic).

If the brush isn't acting on the splats, double-check you're in Edit Mode on the right 3DGS object — Blender's standard texture-paint workflow has overlapping keybinds.

### Step 5 — export the edited splat

1. With the modified 3DGS object selected: `File → Export → 3DGS Render (.ply)`.
2. Name it descriptively — e.g. `young_european_f_anime_v1.ply`, `young_european_f_hair_red.ply`.
3. **Important: export as uncompressed `.ply`**, not compressed. Compressed PLY drops the higher spherical-harmonic bands, which the animation pipeline still uses.
4. Send us back: (a) the exported `.ply`, (b) a Blender viewport screenshot showing what you did, (c) a one-line description of the edit ("retinted hair to teal", "projected anime portrait", "extreme saturation push").

---

## Optional: Claude Desktop driving Blender via MCP

This is the "wow" path — Blender editing by typing into chat. Sometimes faster than hunting through menus, sometimes worse than hands. Worth trying.

### Step A — install `blender-mcp`

1. In Claude Desktop, open `Claude → Settings → Developer → Edit Config`. This opens `claude_desktop_config.json` in an editor.
2. Add (or merge into) the `mcpServers` block:
   ```json
   {
     "mcpServers": {
       "blender": {
         "command": "uvx",
         "args": ["blender-mcp"]
       }
     }
   }
   ```
3. Save the file. Quit Claude Desktop completely, then relaunch.
4. In Blender, install the matching `.py` addon from `https://github.com/ahujasid/blender-mcp` — there's a `BlenderMCP` addon file in the repo. Install it the same way as KIRI. Enable it.
5. In the Blender 3D viewport's `N` panel, find the **BlenderMCP** sub-panel. Click "Connect to Claude". You should see "Connected" within a second or two.
6. Back in Claude Desktop, you should see a hammer icon on the input bar with Blender tools listed. If not, check the config file syntax and restart Claude.

### Step B — drive it

Some prompts to try, in increasing ambition. Paste them into Claude Desktop with Blender running and the splat loaded:

- *"Take a screenshot of the Blender viewport and describe what you see."* (sanity check — confirms scene introspection works)
- *"List all 3D objects in the current Blender scene with their types and positions."*
- *"In the 3DGS panel, push saturation to +0.5 and hue shift to -10. Then take a screenshot."*
- *"Find all splats further than 2 metres from the origin and delete them."* (Claude writes Python, runs it via `execute_blender_code`, the splats get deleted)
- *"Mask all splats whose base RGB colour is within 30 of skin tone (R~210, G~170, B~140) and tint them toward anime-pale (+20 saturation, +30 R)."* (the ambitious one — Claude scripts a per-splat colour edit by iterating `bpy.context.active_object.data` directly)

The `execute_blender_code` tool gives Claude **unrestricted Python access to `bpy`**. Save your `.blend` file before unleashing it on anything important. Save often.

If Claude does something useful, ask it to export the result and tell you the path.

---

## Optional: SuperSplat web editor as a second opinion

If Blender is overkill for what you want, SuperSplat is a free in-browser splat editor. Two ways to run it:

- **Hosted (zero install):** `https://superspl.at/editor` — drag the `.ply` in, use Rect / Brush / Picker selectors to crop or recolour, export.
- **Local build (already checked out on our side, in case you want the same version we tested with):** `git clone https://github.com/playcanvas/supersplat && cd supersplat && npm install && npm run build && npm run serve`, then open `http://localhost:3000`. Identical UI, runs offline.

SuperSplat doesn't have a Claude MCP integration — it's UI-only — but it's the easiest "just clean up / recolour the splats" tool and a great visual sanity check for `.ply` files before and after editing. We confirmed the package's two `.ply` files load cleanly into a local SuperSplat build of v2.25.1.

---

## Quick FAQ

**Q: Can I break the file?**
A: No, the originals are in this folder. Just keep your edited versions named distinctly (`*_v1.ply`, `*_v2.ply`...).

**Q: Will my edits "look right" when animated?**
A: Colour edits → yes, they ride through animation cleanly because expression is a deterministic vertex deformation on top of the splat. Geometry edits (moving splats around) → mostly yes, but if you push them far from the underlying FLAME mesh the per-frame deformation may distort them. Worth trying.

**Q: What spherical harmonic bands does the model use?**
A: **DC only** in this dump. LAM-20K's canonical save writes `f_dc_0..2` (base colour) but skips the higher `f_rest_*` bands, so there's no view-dependent reflectance to worry about — what you see in the viewport from any angle is exactly what the colour is. Brush / texture edits write to `f_dc_*` directly.

**Q: What's the file format actually?**
A: Standard 3DGS PLY layout (Inria-style). Properties per splat in this dump (17 total): `x`, `y`, `z`, `nx`, `ny`, `nz` (zeros), `f_dc_0..2`, `opacity`, `scale_0..2`, `rot_0..3`. ~20k splats. Any 3DGS tool — Blender via KIRI, SuperSplat, Postshot, Polycam, gsplat viewers — can read it.

**Q: How do I check the result before sending?**
A: Re-import the exported `.ply` into a fresh Blender scene (or open it in SuperSplat). If it loads and you can see your edit at viewport-quality, it'll be fine on our end.

---

## What we'll do with it (on our end)

We patch LAM inference to accept a pre-edited canonical `.ply` instead of recomputing one from the anchor PNG, then re-run the exact same 20-second iPhone take with your splat. You get back `take2__<anchor>__<your_edit_name>.mp4` — same camera, same motion, your edit — for direct A/B against the "before" clip in this package.

Success criterion: "this looks more like the stylized target than the baseline did." There's no correct answer; we're prospecting. The interesting failure modes — colour washing out, edits drifting under motion, splats popping — are also useful signal.

Have fun.
