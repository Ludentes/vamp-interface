"""Load the exported Koban canonical mesh: a UV-textured triangle mesh plus
the ARKit-52 verification list.

Canonical source: exp_output/chibi_meshes/koban/Koban Chibi Base Mesh VRM export.blend
Mesh: 5662 verts, 10980 triangle polys.

NOTE ON UV LAYOUT: The Koban OBJ exports a koban_prep.py-generated 'UVMap'
layer. The face island (ARKit-displaced verts) is packed into [0, 0.95]²;
all body loops collapse to the single texel (0.98, 0.98). UV baking (Task 5)
will replace this with a full-coverage FLAME-UV layout.

NOTE ON FACES: The Blender OBJ exporter writes triangle faces (3 vertices per
f-line) for this mesh. The fan-triangulation path in load_koban handles both
tri (3-token) and quad (4-token) f-lines, so it is robust to both.
Face tokens are `v/vt/vn` format; we extract v (index 0) and vt (index 1).

NOTE ON load_flame_uv: We do NOT reuse load_flame_uv here because:
  1. The Koban UV topology is non-injective — many vertices share vt index 1
     (body mesh), which violates load_flame_uv's vt2v uniqueness assertion.
  2. Face count and topology differ from the FLAME template.
  We parse v and vt directly instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from chibi.camera_rig import View, _look_at_w2c

ARKIT_52 = (
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft", "eyeBlinkRight",
    "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight",
    "eyeSquintRight", "eyeWideRight", "jawForward", "jawLeft", "jawRight",
    "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft",
    "mouthRight", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
)


@dataclass
class KobanMesh:
    verts: torch.Tensor      # (V,3) float32
    faces: torch.Tensor      # (F,3) int64  — triangulated (fan-split quads)
    uv: torch.Tensor         # (Nvt,2) float32  — raw vt list from OBJ
    uv_faces: torch.Tensor   # (F,3) int64  — 0-based vt indices, same F as faces


def load_koban(canon_dir: str | Path) -> KobanMesh:
    """Load koban.obj from the canonical dir.

    Parsing strategy:
    - 'v' lines  → vertex positions (V,3)
    - 'vt' lines → UV coords (Nvt,2)
    - 'f' lines  → face tokens in v/vt/vn format, 3 or 4 tokens per face.
      Quads are fan-triangulated into two triangles: (0,1,2) and (0,2,3).

    UVs may extend slightly outside [0,1] due to Blender export precision;
    we do NOT clamp them here — the assertion in the test allows max ≤ 1.0001.
    """
    obj = Path(canon_dir) / "koban.obj"
    verts: list[list[float]] = []
    uvs: list[list[float]] = []
    faces_v: list[list[int]] = []
    faces_vt: list[list[int]] = []

    for ln in obj.read_text().splitlines():
        p = ln.split()
        if not p:
            continue
        if p[0] == "v":
            verts.append([float(p[1]), float(p[2]), float(p[3])])
        elif p[0] == "vt":
            uvs.append([float(p[1]), float(p[2])])
        elif p[0] == "f":
            tokens = p[1:]  # 3 or 4 tokens, each "v/vt/vn" or "v/vt" or "v"
            vs, vts = [], []
            for tok in tokens:
                parts = tok.split("/")
                vs.append(int(parts[0]) - 1)  # 1-based → 0-based
                vt_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else 0
                vts.append(vt_idx)
            # Fan-triangulate: (0,1,2), (0,2,3), (0,3,4), ...
            for i in range(1, len(vs) - 1):
                faces_v.append([vs[0], vs[i], vs[i + 1]])
                faces_vt.append([vts[0], vts[i], vts[i + 1]])

    return KobanMesh(
        verts=torch.tensor(verts, dtype=torch.float32),
        faces=torch.tensor(faces_v, dtype=torch.int64),
        uv=torch.tensor(uvs, dtype=torch.float32),
        uv_faces=torch.tensor(faces_vt, dtype=torch.int64),
    )


# ---------------------------------------------------------------------------
# Frontal view (azim=0, elev=0)
# ---------------------------------------------------------------------------

# Distance derived from the VRM-export mesh bbox (recentred head).
# Head half-extent in X = 0.918; fov=40°, tan(20°) = 0.364.
# dist = 0.918 / tan(20°) * 1.15 (15% margin) ≈ 2.90 → rounded to 3.0.
_FRONTAL_DIST = 3.0
_FRONTAL_FOV_DEG = 40.0
_FRONTAL_IMAGE_SIZE = 512


def make_frontal_view(dist: float = _FRONTAL_DIST,
                      fov_deg: float = _FRONTAL_FOV_DEG,
                      image_size: int = _FRONTAL_IMAGE_SIZE) -> View:
    """Frontal view: azim=0, elev=0, looking at origin."""
    return View(
        w2c=_look_at_w2c(0.0, 0.0, dist),
        fov_rad=math.radians(fov_deg),
        image_size=image_size,
    )


def load_koban_view(canon_dir: str | Path) -> View:
    """Load the frontal View from view.json in canon_dir."""
    import json
    d = json.loads((Path(canon_dir) / "view.json").read_text())
    w2c = torch.tensor(d["w2c"], dtype=torch.float32)
    return View(w2c=w2c, fov_rad=d["fov_rad"], image_size=d["image_size"])


def load_koban_landmarks(
    canon_dir: str | Path,
    return_idx: bool = False,
) -> torch.Tensor:
    """Load landmark data from landmarks.json.

    If return_idx=False (default): returns (K,2) float32 tensor of frontal
    pixel coordinates, obtained by projecting the stored 3D vertex positions
    through the frontal View.

    If return_idx=True: returns (K,) int64 tensor of insightface-106 landmark
    indices corresponding to each landmark.

    landmarks.json schema:
      { "landmarks": [
          { "vertex_idx": int,   # 0-based OBJ vertex index
            "label": str,        # anatomical name
            "insightface_106": int,  # matching IF-106 index
            "pos_world": [x, y, z]   # raw OBJ world-space position
          }, ...
        ]
      }

    pos_world values are in raw OBJ space (head at Y~3.9). A centroid.json
    file (written by koban_prep.py) stores the face-vert centroid; we subtract
    it to recentre before projecting through the origin-centred View.
    """
    import json
    data = json.loads((Path(canon_dir) / "landmarks.json").read_text())
    entries = data["landmarks"]

    if return_idx:
        return torch.tensor(
            [e["insightface_106"] for e in entries], dtype=torch.int64
        )

    # Load centroid and recentre: pos_world is in raw OBJ space; the frontal
    # View expects the mesh recentred on the origin.
    centroid_path = Path(canon_dir) / "centroid.json"
    centroid_data = json.loads(centroid_path.read_text())
    centroid = torch.tensor(centroid_data["centroid"], dtype=torch.float32)

    pts = torch.tensor(
        [e["pos_world"] for e in entries], dtype=torch.float32
    )  # (K,3) raw OBJ space
    pts = pts - centroid  # recentre to origin

    view = load_koban_view(canon_dir)
    px = _project_verts(pts, view)  # (K,2)

    # Sanity-check: all projected pixels must fall within the image
    size = view.image_size
    if not (px.ge(0).all() and px.le(size).all()):
        bad = [(float(px[i, 0]), float(px[i, 1])) for i in range(px.shape[0])
               if float(px[i, 0]) < 0 or float(px[i, 0]) > size
               or float(px[i, 1]) < 0 or float(px[i, 1]) > size]
        raise ValueError(
            f"load_koban_landmarks: {len(bad)} landmark(s) projected outside "
            f"[0, {size}]: {bad[:4]}"
        )
    return px


# ---------------------------------------------------------------------------
# Vertex → pixel projection helpers
# ---------------------------------------------------------------------------

def _project_verts(pts: torch.Tensor, view: View) -> torch.Tensor:
    """Project (K,3) world-space points through a View to (K,2) pixel coords.

    Uses the 3DGS row-vector convention stored in View.w2c:
        p_cam_h = p_world_h @ w2c.T
    (w2c stores the matrix column-major; transposing gives the row-vector form.)
    Then perspective-divides and maps NDC to pixel space (top-left origin).
    """
    K = pts.shape[0]
    ones = torch.ones(K, 1, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=1)            # (K,4)
    p_cam_h = pts_h @ view.w2c.T                     # (K,4)  p_world_h @ w2c.T
    p_cam = p_cam_h[:, :3] / p_cam_h[:, 3:4]        # (K,3) perspective divide

    if (p_cam[:, 2] <= 0).any():
        n_behind = int((p_cam[:, 2] <= 0).sum())
        raise ValueError(
            f"_project_verts: {n_behind} point(s) are behind the camera "
            f"(cam-Z <= 0). Check that the mesh is recentred and the camera "
            f"is in front of it."
        )

    # Perspective projection: fx = fy = 1/tan(fov/2), principal point = (0,0)
    f = 1.0 / math.tan(view.fov_rad / 2.0)
    x_ndc = p_cam[:, 0] / (p_cam[:, 2] * f) * (-1)  # flip X: cam-right is screen-left in OpenGL
    y_ndc = p_cam[:, 1] / (p_cam[:, 2] * f) * (-1)  # flip Y: cam-down is screen-up

    # NDC [-1,1] → pixel [0, image_size]
    size = view.image_size
    px = (x_ndc + 1.0) * 0.5 * size
    py = (y_ndc + 1.0) * 0.5 * size
    return torch.stack([px, py], dim=1)              # (K,2)
