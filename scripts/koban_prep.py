"""Run inside Blender: blender -b <koban.blend> --python scripts/koban_prep.py -- <outdir>

Exports the 'Chibi Base Mesh' object to OBJ with UVs, and dumps the ARKit-52
shape-key names to arkit_keys.json so koban_asset.py can verify them.

Probed object name: "Chibi Base Mesh" (confirmed from Koban Chibi Base Mesh 1.0.blend).
The mesh has 5664 verts, 5674 polys, 3 UV layers, 61 shape keys (52 ARKit + extras).
"""
import sys
import os
import json

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
outdir = argv[0]
os.makedirs(outdir, exist_ok=True)

MESH_OBJ_NAME = "Chibi Base Mesh"

obj = bpy.data.objects[MESH_OBJ_NAME]
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
