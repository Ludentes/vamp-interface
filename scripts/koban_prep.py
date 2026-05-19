"""Run inside Blender: blender -b <koban.blend> --python scripts/koban_prep.py -- <outdir>
Generates a frontal-projection UV unwrap for the Koban chibi mesh (it ships
none), exports it to OBJ, and dumps the shape-key names to arkit_keys.json."""
import sys, os, json
import bpy

# Must stay in sync with src/chibi/koban_asset.py:ARKIT_52
# The 52 ARKit blendshape names — used to derive the face-region mask.
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
EPS = 1e-4          # vertex-displacement threshold for "is face vertex"
FACE_RECT = 0.95    # face UV island occupies [0, FACE_RECT]^2
SKIN_TEXEL = 0.98   # non-face loops collapse to this UV

argv = sys.argv[sys.argv.index("--") + 1:]
outdir = argv[0]
os.makedirs(outdir, exist_ok=True)

obj = bpy.data.objects["Chibi Base Mesh"]
me = obj.data
kb = me.shape_keys.key_blocks

with open(os.path.join(outdir, "arkit_keys.json"), "w") as f:
    json.dump([k.name for k in kb], f, indent=2)

# --- face mask: verts displaced by any ARKit-52 key relative to Basis ---
basis = kb["Basis"].data
nv = len(me.vertices)
is_face_vert = [False] * nv
for name in ARKIT_52:
    if name not in kb:
        continue
    skd = kb[name].data
    for i in range(nv):
        if (skd[i].co - basis[i].co).length > EPS:
            is_face_vert[i] = True
face_poly = {p.index for p in me.polygons
             if any(is_face_vert[v] for v in p.vertices)}

# --- frontal bbox of face verts (x = left-right, z = up; y = depth) ---
fvi = [i for i in range(nv) if is_face_vert[i]]
# fvi_set used later to compute the OBJ-space centroid after export
fvi_set = set(fvi)

xs = [me.vertices[i].co.x for i in fvi]
zs = [me.vertices[i].co.z for i in fvi]
xmin, zmin = min(xs), min(zs)
xr = (max(xs) - xmin) or 1.0
zr = (max(zs) - zmin) or 1.0

# --- write the UV layer (clear existing layers first) ---
while me.uv_layers:
    me.uv_layers.remove(me.uv_layers[0])
uvl = me.uv_layers.new(name="UVMap")
for p in me.polygons:
    is_face = p.index in face_poly
    for li in range(p.loop_start, p.loop_start + p.loop_total):
        if is_face:
            co = me.vertices[me.loops[li].vertex_index].co
            u = max(0.0, min(FACE_RECT, (co.x - xmin) / xr * FACE_RECT))
            v = max(0.0, min(FACE_RECT, (co.z - zmin) / zr * FACE_RECT))
            uvl.data[li].uv = (u, v)
        else:
            uvl.data[li].uv = (SKIN_TEXEL, SKIN_TEXEL)
me.update()

# --- export ---
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
obj_path = os.path.join(outdir, "koban.obj")
bpy.ops.wm.obj_export(filepath=obj_path,
                      export_selected_objects=True, export_uv=True,
                      export_normals=True, export_materials=False,
                      apply_modifiers=False)

# Compute face-vert centroid from the exported OBJ (correct OBJ coordinate space).
# Blender's internal vertex.co is in Blender's Z-up space; the OBJ exporter
# applies a coordinate transform, so we must read back OBJ positions, not use
# vertex.co directly. Blender vertex order is preserved in OBJ output.
obj_verts = []
with open(obj_path) as fh:
    for line in fh:
        if line.startswith("v "):
            parts = line.split()
            obj_verts.append((float(parts[1]), float(parts[2]), float(parts[3])))

face_verts_obj = [obj_verts[i] for i in fvi_set if i < len(obj_verts)]
n_fv = len(face_verts_obj)
cx = sum(p[0] for p in face_verts_obj) / n_fv
cy = sum(p[1] for p in face_verts_obj) / n_fv
cz = sum(p[2] for p in face_verts_obj) / n_fv

with open(os.path.join(outdir, "centroid.json"), "w") as f:
    json.dump({"centroid": [round(cx, 6), round(cy, 6), round(cz, 6)],
               "_meta": {"source": "mean of is_face_vert vertices in OBJ world space "
                                   "(read back from exported OBJ)",
                         "n_face_verts": n_fv}}, f, indent=2)

print(f"[koban_prep] face polys {len(face_poly)}/{len(me.polygons)}; "
      f"centroid {cx:.3f},{cy:.3f},{cz:.3f}; "
      f"exported koban.obj + arkit_keys.json + centroid.json to {outdir}")
