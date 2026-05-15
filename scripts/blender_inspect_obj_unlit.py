"""Headless Blender — render with the Workbench engine in 'Vertex Color'
shading mode. This is what the artist sees in the viewport with shading
mode set to Solid and Color override = Attribute (default per-vertex).
"""
import sys, os, argparse

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--obj", required=True)
ap.add_argument("--outdir", required=True)
args = ap.parse_args(argv)

import bpy, mathutils

os.makedirs(args.outdir, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

bpy.ops.wm.obj_import(filepath=args.obj)
obj = bpy.context.selected_objects[0]
me = obj.data
color_attr = next((a for a in me.color_attributes), None)
print(f"[blender] verts={len(me.vertices)} polys={len(me.polygons)}  color_attr={color_attr.name if color_attr else None}")

bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
cx = sum(v.x for v in bbox) / 8; cy = sum(v.y for v in bbox) / 8; cz = sum(v.z for v in bbox) / 8
extent = max(max(v.x for v in bbox) - min(v.x for v in bbox),
             max(v.y for v in bbox) - min(v.y for v in bbox),
             max(v.z for v in bbox) - min(v.z for v in bbox))

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.color_type = "VERTEX"   # Solid mode with vertex color
scene.display.shading.light = "FLAT"          # No lighting — pure unlit colour
scene.render.resolution_x = 768
scene.render.resolution_y = 768
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"

for name, loc in [
    ("unlit_front", (cx, cy - extent * 2.2, cz)),
    ("unlit_three_quarter", (cx + extent * 1.5, cy - extent * 1.8, cz + extent * 0.6)),
]:
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.object
    bpy.ops.object.empty_add(location=(cx, cy, cz))
    tgt = bpy.context.object
    tc = cam.constraints.new("TRACK_TO"); tc.target = tgt
    tc.track_axis = "TRACK_NEGATIVE_Z"; tc.up_axis = "UP_Y"
    scene.camera = cam
    scene.render.filepath = os.path.join(args.outdir, f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[blender] wrote {scene.render.filepath}")
    bpy.data.objects.remove(cam, do_unlink=True)
    bpy.data.objects.remove(tgt, do_unlink=True)
