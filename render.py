import bpy
from blender_addon_proxy_scene import build_setup
import numpy as np
from pathlib import Path
import gsoup


def almost_black(image_path):
    image = gsoup.load_image(image_path)
    test = image.mean(axis=-1)
    total_pixels = test.shape[0] * test.shape[1]
    active_pixels = np.count_nonzero(test > 1)
    dead_pixels = total_pixels - active_pixels
    if (dead_pixels / total_pixels) > 0.8:
        is_all_black = True
    else:
        is_all_black = False
    return is_all_black


def render_direct_light(depth_image_path, dst_path, randcolor=False, hints=False, test_almost_black=False):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath="./booth.blend")
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    # for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    #     device.use = True
    depth_image = bpy.data.images.load(
            filepath=str(depth_image_path),
        )
    nodetree = {
    "depthimage": depth_image,
    "image": None,
    "sna_normalempty": None,
    "aspect_ratio": float(depth_image.size[0] / depth_image.size[1]),
    "uses_rgb_image": False,
    "emitting_material": False
    }
    build_setup(nodetree)
    pseudo_scene = bpy.data.objects["pseudo_scene"]
    material = pseudo_scene.active_material
    material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0
    material.node_tree.nodes["Principled BSDF"].inputs["Specular"].default_value = 0.0
    dir_output_node = bpy.data.scenes["Scene"].node_tree.nodes["render_dir_png"]
    is_black = True
    while is_black:
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-0.5, 0.0)
        z = np.random.uniform(-0.5, 0.5)
        bpy.data.objects["Light"].location = (x, y, z)
        if randcolor:
            bpy.data.objects["Light"].data.color = np.random.rand(3)  # np.random.normal(1.0, 0.5, size=(3,))
        bpy.data.objects["Light"].data.energy = 10
        dir_output_node.base_path = str(dst_path.parent)
        dir_output_node.file_slots[0].path = str(dst_path.stem)
        bpy.ops.render.render()
        if test_almost_black:
            tmp_output_blender = Path(dst_path.parent, dst_path.stem + "0000.png")
            if not almost_black(tmp_output_blender):
                is_black = False
        else:
            is_black = False

    if hints:
        spec = [1.0, 1.0, 1.0]
        roughness = [0.05, 0.13, 0.34]
        for i in range(3):
            # material = bpy.context.pseudo_scene.active_material
            material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = roughness[i]
            material.node_tree.nodes["Principled BSDF"].inputs["Specular"].default_value = spec[i]
            cur_hint_path = Path(dst_path.parent, dst_path.stem + "hint_{:02d}_".format(i) + dst_path.suffix)
            dir_output_node.base_path = str(cur_hint_path.parent)
            dir_output_node.file_slots[0].path = str(cur_hint_path.stem)
            bpy.ops.render.render()
    bpy.data.objects.remove(pseudo_scene)

if __name__ == "__main__":
    src = Path("output/inference/tokyo_manual/disp_norm_00_00_00.png")
    for i in range(5):
        dst = Path("renders", "{:02}_.png".format(i))
        render_direct_light(src, dst)