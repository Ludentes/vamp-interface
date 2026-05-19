"""blender -b <koban.blend> --python scripts/koban_rig_test.py -- <outdir>
Spike S2: drive the Koban mesh's native ARKit-52 shape keys and render
front views to confirm the rig deforms geometry cleanly."""
import sys, os, math
import bpy
from mathutils import Vector

outdir = sys.argv[sys.argv.index("--") + 1:][0]
os.makedirs(outdir, exist_ok=True)

obj = bpy.data.objects["Chibi Base Mesh"]
kb = obj.data.shape_keys.key_blocks

# Hide armatures and empties from render so rig widgets don't pollute the frame
for o in bpy.data.objects:
    if o.type in ("ARMATURE", "EMPTY"):
        o.hide_render = True

# Compute world-space bounding box of the head mesh
mn = Vector((1e9, 1e9, 1e9))
mx = Vector((-1e9, -1e9, -1e9))
for corner in obj.bound_box:
    w = obj.matrix_world @ Vector(corner)
    for i in range(3):
        mn[i] = min(mn[i], w[i])
        mx[i] = max(mx[i], w[i])
center = (mn + mx) / 2
size = max(mx - mn)  # longest axis

# Camera: front-on view, distance = 1.7 × size
cam_data = bpy.data.cameras.new("RigTestCam")
cam_obj = bpy.data.objects.new("RigTestCam", cam_data)
bpy.context.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

dist = size * 1.7
# Front view: Y-negative direction (Blender +Y = back)
cam_obj.location = Vector((center.x, center.y - dist, center.z + size * 0.1))
direction = (center - cam_obj.location).normalized()
cam_obj.rotation_euler = (
    math.atan2(math.hypot(direction.x, direction.y), direction.z),
    0.0,
    math.atan2(-direction.x, direction.y),
)

# Lighting: sun lamp + world background
sun_data = bpy.data.lights.new("RigTestSun", "SUN")
sun_data.energy = 3.0
sun_obj = bpy.data.objects.new("RigTestSun", sun_data)
bpy.context.collection.objects.link(sun_obj)
sun_obj.rotation_euler = (math.radians(55), 0.0, math.radians(40))

world = bpy.data.worlds.new("RigTestWorld")
bpy.context.scene.world = world
world.use_nodes = True
bg_node = world.node_tree.nodes["Background"]
bg_node.inputs[0].default_value = (0.85, 0.86, 0.9, 1.0)
bg_node.inputs[1].default_value = 0.6

# Render settings
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 720
scene.render.resolution_y = 900
scene.render.film_transparent = False

# Shape key test cases
tests = {
    "neutral":  [],
    "blink":    ["eyeBlinkLeft", "eyeBlinkRight"],
    "jawOpen":  ["jawOpen"],
    "smile":    ["mouthSmileLeft", "mouthSmileRight"],
}

kb_names = {k.name for k in kb}
print(f"[rig_test] {len(kb)} shape keys found in mesh")

for label, keys in tests.items():
    # Reset all shape keys to 0
    for k in kb:
        k.value = 0.0
    # Drive the target keys
    for name in keys:
        if name in kb_names:
            kb[name].value = 1.0
        else:
            print(f"[rig_test] WARNING: shape key '{name}' not found — skipping")
    # Render
    scene.render.filepath = os.path.join(outdir, f"rig_{label}.png")
    bpy.ops.render.render(write_still=True)
    print(f"[rig_test] rendered {label}")

print("[rig_test] done")
