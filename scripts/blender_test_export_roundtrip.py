"""Test: import OBJ → modify vertex colours → export OBJ → verify the export
file has `v x y z r g b` rows so LAM's hook can read it.

Also exercise vertex-colour smoothing to see whether that's a reasonable
artist first step.
"""
import sys, os, argparse

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--outdir", required=True)
args = ap.parse_args(argv)

import bpy

os.makedirs(args.outdir, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.wm.obj_import(filepath=args.input)
obj = bpy.context.selected_objects[0]
me = obj.data
ca = me.color_attributes[0]
print(f"[blender] imported  verts={len(me.vertices)}  color_attr={ca.name}  domain={ca.domain}")

# --- Edit 1: tint top 25% of verts (by y) red, in-place
ys = [v.co.y for v in me.vertices]
y_thresh = sorted(ys)[int(0.75 * len(ys))]
n_tinted = 0
for vi, v in enumerate(me.vertices):
    if v.co.y > y_thresh:
        ca.data[vi].color = (0.85, 0.10, 0.10, 1.0)
        n_tinted += 1
print(f"[blender] tinted {n_tinted}/{len(me.vertices)} verts above y={y_thresh:.3f}")

# --- Export with vertex colours ---
out_path = os.path.join(args.outdir, "exported_red_hair_blender.obj")
try:
    bpy.ops.wm.obj_export(
        filepath=out_path,
        export_selected_objects=True,
        export_colors=True,
        export_normals=False,
        export_uv=False,
        export_materials=False,
        export_triangulated_mesh=False,
        path_mode="AUTO",
    )
except TypeError as e:
    print(f"[blender] obj_export call failed with named args: {e}")
    bpy.ops.wm.obj_export(filepath=out_path, export_colors=True)
print(f"[blender] wrote {out_path}")

# --- Inspect what Blender exported ---
with open(out_path) as f:
    header_lines = [next(f) for _ in range(5)]
    f.seek(0)
    v_lines = [L for L in f.readlines() if L.startswith("v ")]
print(f"[blender] export: {len(v_lines)} 'v' lines; first one:")
print(f"          {v_lines[0].strip()}")
parts = v_lines[0].split()
print(f"          {len(parts)} columns → {'v x y z r g b' if len(parts) == 7 else 'NOT 7 cols, MISMATCH'}")
