"""Headless Blender script: import a vertex-coloured OBJ and render three
views so we can verify what the artist will see in their viewport.

Run via:
    ~/local/blender/blender -b -P scripts/blender_inspect_obj.py -- \
        --obj <path.obj> --outdir <dir>
"""
import sys, os, argparse, math

# Parse args after the `--`
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--obj", required=True)
ap.add_argument("--outdir", required=True)
args = ap.parse_args(argv)

import bpy

os.makedirs(args.outdir, exist_ok=True)

# Wipe default scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# Import OBJ. Blender 4.x wm.obj_import handles extended Wavefront with vertex colours.
bpy.ops.wm.obj_import(filepath=args.obj)
obj = bpy.context.selected_objects[0]
print(f"[blender] imported: {obj.name}  verts={len(obj.data.vertices)}  polys={len(obj.data.polygons)}")
me = obj.data
# Probe colour attribute
color_attr_name = None
for attr in me.color_attributes:
    color_attr_name = attr.name
    print(f"[blender]   color attr: name='{attr.name}'  domain={attr.domain}  type={attr.data_type}  count={len(attr.data)}")
if color_attr_name is None:
    print("[blender]   NO COLOR ATTRIBUTE FOUND on imported mesh")

# Centre + frame
import mathutils
# Recompute bbox
bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
cx = sum(v.x for v in bbox) / 8
cy = sum(v.y for v in bbox) / 8
cz = sum(v.z for v in bbox) / 8
ext_x = max(v.x for v in bbox) - min(v.x for v in bbox)
ext_y = max(v.y for v in bbox) - min(v.y for v in bbox)
ext_z = max(v.z for v in bbox) - min(v.z for v in bbox)
extent = max(ext_x, ext_y, ext_z)
print(f"[blender]   bbox extent={extent:.3f}  centre=({cx:.3f},{cy:.3f},{cz:.3f})")

# Material that reads vertex colour as Base Color → looks like the artist's
# "Material Preview" mode in viewport.
if color_attr_name:
    mat = bpy.data.materials.new(name="VertexColorMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    attr_node = nodes.new("ShaderNodeVertexColor")
    attr_node.layer_name = color_attr_name
    mat.node_tree.links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(mat)

# Add light + cam
bpy.ops.object.light_add(type="SUN", location=(0, -extent * 3, extent * 2))
bpy.context.object.data.energy = 3.0
bpy.ops.object.light_add(type="SUN", location=(extent * 2, extent * 2, extent * 2))
bpy.context.object.data.energy = 1.0

# Render settings
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x = 768
scene.render.resolution_y = 768
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"

# Three cameras: front, 3/4, top-down — what the artist sees on import.
cam_setups = [
    ("front",  (cx, cy - extent * 2.2, cz)),
    ("three_quarter", (cx + extent * 1.5, cy - extent * 1.8, cz + extent * 0.6)),
    ("top_down", (cx, cy, cz + extent * 2.5)),
]
for name, loc in cam_setups:
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.object
    # Track toward centre
    bpy.ops.object.empty_add(location=(cx, cy, cz))
    target = bpy.context.object
    tc = cam.constraints.new("TRACK_TO")
    tc.target = target
    tc.track_axis = "TRACK_NEGATIVE_Z"
    tc.up_axis = "UP_Y"
    scene.camera = cam
    scene.render.filepath = os.path.join(args.outdir, f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[blender] wrote {scene.render.filepath}")
    # Clean up cam + target so the next loop has a clean slate
    bpy.data.objects.remove(cam, do_unlink=True)
    bpy.data.objects.remove(target, do_unlink=True)

# Also save a wireframe-only render to show the topology the artist sees in Edit Mode.
# Use freestyle for a clean wireframe.
scene.render.use_freestyle = True
scene.view_layers[0].freestyle_settings.crease_angle = 3.14
# Hide the material so freestyle pops
scene.render.filepath = os.path.join(args.outdir, "wireframe.png")
# Re-add front cam
bpy.ops.object.camera_add(location=(cx, cy - extent * 2.2, cz))
cam = bpy.context.object
bpy.ops.object.empty_add(location=(cx, cy, cz))
target = bpy.context.object
tc = cam.constraints.new("TRACK_TO")
tc.target = target
tc.track_axis = "TRACK_NEGATIVE_Z"
tc.up_axis = "UP_Y"
scene.camera = cam
# White material to make freestyle lines visible
white = bpy.data.materials.new("White")
white.use_nodes = True
bsdf = white.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
obj.data.materials.clear()
obj.data.materials.append(white)
bpy.ops.render.render(write_still=True)
print(f"[blender] wrote {scene.render.filepath}")

print("[blender] done")
