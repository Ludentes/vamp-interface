"""Run inside Blender: blender -b <koban.blend> --python scripts/koban_prep.py -- <outdir>

Exports the 'Chibi Base Mesh' object to OBJ with UVs, and dumps the ARKit-52
shape-key names to arkit_keys.json so koban_asset.py can verify them.

Canonical source: exp_output/chibi_meshes/koban/Koban Chibi Base Mesh VRM export.blend
Probed object name: "Chibi Base Mesh", 5662 verts, 10980 triangle polys.
UV layers: UVMap (active, proper full-body unwrap), UV0, UVMap.001.
Shape keys: 61 total including full camelCase ARKit-52 set plus Basis + extras.

The script explicitly sets UVMap as the active UV layer before export to ensure
the correct unwrap is written (not UV0 or UVMap.001).
"""
import sys
import os
import json

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
outdir = argv[0]
os.makedirs(outdir, exist_ok=True)

MESH_OBJ_NAME = "Chibi Base Mesh"
UV_LAYER_NAME = "UVMap"

obj = bpy.data.objects[MESH_OBJ_NAME]

# Ensure the correct UV layer is active before export
uv_layers = obj.data.uv_layers
if UV_LAYER_NAME in uv_layers:
    uv_layers.active = uv_layers[UV_LAYER_NAME]
    print(f"[koban_prep] set active UV layer: {UV_LAYER_NAME}")
else:
    available = [l.name for l in uv_layers]
    print(f"[koban_prep] WARNING: UV layer '{UV_LAYER_NAME}' not found; available: {available}")

active_uv = obj.data.uv_layers.active
print(f"[koban_prep] active UV layer name: {active_uv.name!r}, loop count: {len(active_uv.data)}")

keys = [k.name for k in obj.data.shape_keys.key_blocks] if obj.data.shape_keys else []
with open(os.path.join(outdir, "arkit_keys.json"), "w") as f:
    json.dump(keys, f, indent=2)

bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.wm.obj_export(
    filepath=os.path.join(outdir, "koban.obj"),
    export_selected_objects=True,
    export_uv=True,
    export_normals=True,
    export_materials=False,
    apply_modifiers=False,
)
print(f"[koban_prep] exported koban.obj + arkit_keys.json to {outdir}")
