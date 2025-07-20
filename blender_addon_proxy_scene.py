import bpy

# adapted addon from https://www.immersity.ai
bl_info = {
    "name": "PseudoScene",
    "author": "yoterel",
    "description": "",
    "blender": (3, 3, 0),
    "version": (1, 1, 2),
    "location": "Import",
    "warning": "",
    "doc_url": "",
    "tracker_url": "",
    "category": "Import-Export",
}


import bpy
import bpy.utils.previews
import math
import os
from bpy_extras.io_utils import ImportHelper, ExportHelper

# class PS_Preferences(bpy.types.AddonPreferences):
#     bl_idname = "importdepthmap"

#     def draw(self, context):
#         if not (False):
#             layout = self.layout
#             row = layout.row(heading="", align=False)
#             row.alert = False
#             row.enabled = True
#             row.active = True
#             row.use_property_split = False
#             row.use_property_decorate = False
#             row.scale_x = 1.0
#             row.scale_y = 1.0
#             row.alignment = "Expand".upper()
#             if not True:
#                 row.operator_context = "EXEC_DEFAULT"
#             row.prop(
#                 bpy.context.scene,
#                 "sna_camerasetup",
#                 text="Create Camera Setup",
#                 icon_value=0,
#                 emboss=True,
#             )


# def add_camera():
#     bpy.ops.object.camera_add(
#         "INVOKE_DEFAULT",
#         location=nodetree["sna_normalempty"].location,
#         rotation=(math.radians(90.0), 0.0, 0.0),
#     )
#     nodetree["sna_camera"] = bpy.context.view_layer.objects.active
#     constraint = nodetree["sna_normalempty"].constraints.new(
#         type="COPY_TRANSFORMS",
#     )
#     constraint.target = nodetree["sna_camera"]
#     constraint.influence = 0.5


def add_to_topbar(self, context):
    if not (False):
        layout = self.layout
        op = layout.operator(
            "ps.load_depth_image",
            text="Import Depth Map",
            icon_value=0,
            emboss=True,
            depress=False,
        )
        # op.sna_seperate_depth_map = False
        # op.sna_pano = False


class PS_LoadImage(bpy.types.Operator, ImportHelper):
    bl_idname = "ps.load_image"
    bl_label = "Load Image"
    bl_description = "Load combined color and depth image"
    bl_options = {"REGISTER", "UNDO"}
    filter_glob: bpy.props.StringProperty(
        default="*.png;*.jpg;*.exr", options={"HIDDEN"}
    )
    # sna_pano: bpy.props.BoolProperty(name="360Pano", description="", default=False)

    # @classmethod
    # def poll(cls, context):
    #     if bpy.app.version >= (3, 0, 0) and False:
    #         cls.poll_message_set()
    #     return not False

    def execute(self, context):
        image = bpy.data.images.load(
            filepath=self.filepath,
        )
        nodetree["image"] = image
        build_setup()
        return {"FINISHED"}


class PS_LoadDepth(bpy.types.Operator, ImportHelper):
    bl_idname = "ps.load_depth_image"
    bl_label = "Load Depth Image"
    bl_description = ""
    bl_options = {"REGISTER", "UNDO"}
    filter_glob: bpy.props.StringProperty(
        default="*.png;*.jpg;*.exr", options={"HIDDEN"}
    )
    sna_seperate_depth_map: bpy.props.BoolProperty(
        name="Also Load Color Map",
        description="Select a seperate file to load as texture for the depthmap",
        default=False,
    )
    emitting_material: bpy.props.BoolProperty(
        name="Color Is Emitter",
        description="If true, the objects color will be emitting light using the rgb image",
        default=False,
    )
    # @classmethod
    # def poll(cls, context):
    #     if bpy.app.version >= (3, 0, 0) and False:
    #         cls.poll_message_set()
    #     return not False

    def execute(self, context):
        image = bpy.data.images.load(
            filepath=self.filepath,
        )
        # addon_keymaps = {}
        # _icons = None
        nodetree = {
            "depthimage": None,
            "image": None,
            "sna_normalempty": None,
            # "sna_camera": None,
            # "sna_pano": False,
            "aspect_ratio": 0.0,
            "uses_rgb_image": False,
            "emitting_material": False
        }
        nodetree["uses_rgb_image"] = self.sna_seperate_depth_map
        # nodetree["sna_pano"] = self.sna_pano
        nodetree["emitting_material"] = self.emitting_material
        # nodetree["image"] = image
        nodetree["aspect_ratio"] = float(image.size[0] / image.size[1])
        nodetree["depthimage"] = image
        if nodetree["uses_rgb_image"]:
            bpy.ops.ps.load_image("INVOKE_DEFAULT")
        else:
            build_setup(nodetree)
        return {"FINISHED"}


def build_setup(nodetree):
    # if bpy.context.scene.sna_camerasetup:
    #     if nodetree["sna_pano"]:
    #         pass
    #     else:
    #         bpy.ops.object.empty_add(
    #             "INVOKE_DEFAULT",
    #             type="SINGLE_ARROW",
    #             radius=0.3,
    #             location=(0.0, -2.0, 0.0),
    #             rotation=(math.radians(-90.0), 0.0, 0.0),
    #         )
    #         bpy.context.view_layer.objects.active.name = (
    #             nodetree["image"].name.split(",")[0] + "_Align"
    #         )
    #         nodetree["sna_normalempty"] = bpy.context.view_layer.objects.active
    #         add_camera()
    # if nodetree["sna_pano"]:
    #     bpy.ops.mesh.primitive_uv_sphere_add(
    #         "INVOKE_DEFAULT", segments=32, ring_count=16, radius=50.0
    #     )
    #     bpy.context.view_layer.objects.active.name = nodetree["image"].name.split(
    #         ","
    #     )[0]
    #     bpy.ops.object.shade_smooth(
    #         "INVOKE_DEFAULT", use_auto_smooth=True, auto_smooth_angle=math.radians(60.0)
    #     )
    #     bpy.ops.object.mode_set("INVOKE_DEFAULT", mode="EDIT", toggle=False)
    #     bpy.ops.mesh.flip_normals(
    #         "INVOKE_DEFAULT",
    #     )
    #     bpy.ops.object.mode_set("INVOKE_DEFAULT", mode="OBJECT", toggle=False)
    #     bpy.ops.object.transform_apply(
    #         "INVOKE_DEFAULT", location=True, rotation=True, scale=True
    #     )
    #     material_setup()
    #     bpy.ops.object.modifier_add("INVOKE_DEFAULT", type="SUBSURF")
    #     subdivision_settings(
    #         "SIMPLE Subsurf", 5, "CATMULL_CLARK", 0
    #     )  # LeiaInc LI1 modifications: changed levels from 3 to 5
    #     bpy.ops.object.modifier_add("INVOKE_DEFAULT", type="DISPLACE")
    #     set_displace_settings(1, 50.0)
    #     bpy.ops.object.modifier_add("INVOKE_DEFAULT", type="SUBSURF")
    #     subdivision_settings("Smooth Subsurf", 1, "CATMULL_CLARK", 2)
    # else:
    if not bpy.data.objects.get(
        "Empty_DepthMapReference"
    ):  # LeiaInc LI1 modification: create empty object for Simple Deform Modifier to use
        bpy.ops.object.empty_add(
            type="PLAIN_AXES", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
        )
        bpy.context.view_layer.objects.active.name = "Empty_DepthMapReference"
        bpy.context.object.rotation_euler[2] = (
            3.14159  # rotate by 180 degrees around z axis
        )
        bpy.context.view_layer.objects.active.hide_set(True)
        bpy.context.view_layer.objects.active.hide_render = True
    bpy.ops.mesh.primitive_plane_add("INVOKE_DEFAULT", size=1.0)
    bpy.context.view_layer.objects.active.name = "pseudo_scene"
    bpy.ops.object.shade_smooth(
        "INVOKE_DEFAULT", use_auto_smooth=True, auto_smooth_angle=math.radians(60.0)
    )
    bpy.context.view_layer.objects.active.rotation_euler = (
        math.radians(90.0),
        0.0,
        0.0,
    )
    bpy.context.view_layer.objects.active.location = (0.0, 0.0, 0.5)
    bpy.context.view_layer.objects.active.scale = (
        nodetree["aspect_ratio"],
        1.0,
        1.0,
    )
    bpy.ops.object.transform_apply(
        "INVOKE_DEFAULT", location=True, rotation=True, scale=True
    )
    bpy.ops.object.modifier_add("INVOKE_DEFAULT", type="SUBSURF")
    # LeiaInc LI1 modifications: changed levels from 6 to 8
    set_subdivision_settings(0, 9, "SIMPLE")  
    # LeiaInc LI1 - Add Simple Deform Modifier
    bpy.ops.object.modifier_add("INVOKE_DEFAULT", type="SIMPLE_DEFORM")
    set_deform_settings(1)  # LeiaInc LI1 - Add Simple Deform Modifier
    #bpy.ops.object.modifier_add("INVOKE_DEFAULT", type="NORMAL_EDIT")
    #normal_settings(2)  # LeiaInc LI1 - index incremented 1 to 2
    bpy.ops.object.modifier_add("INVOKE_DEFAULT", type="DISPLACE")
    set_displace_settings(2, 1.0, nodetree)  # LeiaInc LI1 - index incremented 2 to 3
    #bpy.ops.object.modifier_add("INVOKE_DEFAULT", type="SUBSURF")
    #subdivision_settings(
    #    "Smooth Subsurf", 1, "CATMULL_CLARK", 4
    #)  # LeiaInc LI1 - index incremented 3 to 4
    bpy.ops.object.modifier_add("INVOKE_DEFAULT", type="SMOOTH")
    set_smooth_settings(3, 3.0, 5)
    
    material_setup(nodetree)

def set_subdivision_settings(index, levels, sd_type):
    # list(bpy.context.view_layer.objects.active.modifiers)[index].name = Name
    list(bpy.context.view_layer.objects.active.modifiers)[index].subdivision_type = sd_type
    list(bpy.context.view_layer.objects.active.modifiers)[index].render_levels = levels
    list(bpy.context.view_layer.objects.active.modifiers)[index].levels = levels

def set_smooth_settings(index, factor, iterations):
    # list(bpy.context.view_layer.objects.active.modifiers)[index].name = Name
    list(bpy.context.view_layer.objects.active.modifiers)[index].factor = factor
    list(bpy.context.view_layer.objects.active.modifiers)[index].iterations = iterations
    
def set_displace_settings(index, strength, nodetree):
    list(bpy.context.view_layer.objects.active.modifiers)[index].texture_coords = "UV"
    list(bpy.context.view_layer.objects.active.modifiers)[index].direction = "CUSTOM_NORMAL"
    texture = bpy.data.textures.new(
        name="DepthDisplace",
        type="IMAGE",
    )
    list(bpy.context.view_layer.objects.active.modifiers)[index].texture = texture
    list(bpy.context.view_layer.objects.active.modifiers)[index].strength = strength
    texture.extension = "EXTEND"  # LeiaInc LI1 modification: changed from CLIP to EXTEND to avoid wrapping of UV
    # if nodetree["uses_rgb_image"]:
    texture.image = nodetree["depthimage"]
    # else:
    #     texture.image = nodetree["image"]
    #     list(bpy.context.view_layer.objects.active.modifiers)[
    #         index
    #     ].texture.crop_min_x = 0.5
    texture.image.colorspace_settings.name = "Raw"  # LeiaInc LI1 modification: change from sRGB (default)_to Raw so that colorspace mapping does not skew depth map

def set_deform_settings(index):  # LeiaInc LI1 - Add Simple Deform Modifier
    list(bpy.context.view_layer.objects.active.modifiers)[index].origin = (
        bpy.data.objects["Empty_DepthMapReference"]
    )
    list(bpy.context.view_layer.objects.active.modifiers)[index].deform_method = "BEND"
    list(bpy.context.view_layer.objects.active.modifiers)[index].deform_axis = "Z"
    # default to no curvature
    list(bpy.context.view_layer.objects.active.modifiers)[index].angle = 0

def set_normal_settings(index, nodetree):
    bpy.context.active_object.modifiers[index].target = nodetree["sna_normalempty"]
    list(bpy.context.view_layer.objects.active.modifiers)[index].mode = "DIRECTIONAL"

def material_setup(nodetree):
    bpy.ops.object.material_slot_add(
        "INVOKE_DEFAULT",
    )
    material = bpy.data.materials.new(
        name="my_material",  # nodetree["image"].name
    )
    material.use_nodes = True
    bpy.context.object.active_material = material
    material.node_tree.nodes["Principled BSDF"].inputs[
        "Roughness"
    ].default_value = 1.0
    material.node_tree.nodes["Principled BSDF"].inputs[
        "Specular"
    ].default_value = 0.0
    # material.node_tree.nodes["Principled BSDF"].inputs[
    #     "Base Color"
    # ].default_value = (1, 1, 1, 1)
    if nodetree["uses_rgb_image"]:
        node_0 = material.node_tree.nodes.new(
            type="ShaderNodeTexImage",
        )
        node_0.location = (-300.0, 100.0)
        node_0.image = nodetree["image"]
    
        link_0 = material.node_tree.links.new(
            input=material.node_tree.nodes["Principled BSDF"].inputs[0],
            output=node_0.outputs[0],
        )
        if nodetree["emitting_material"]:
            link_1 = material.node_tree.links.new(
                input=material.node_tree.nodes["Principled BSDF"].inputs["Emission"],
                output=node_0.outputs[0],
            )
    # if nodetree["uses_rgb_image"]:
    #     pass
    # else:
    #     node_1 = material.node_tree.nodes.new(
    #         type="ShaderNodeMapping",
    #     )
    #     node_1.location = (-500.0, 100.0)
    #     link_2 = material.node_tree.links.new(
    #         input=node_0.inputs[0],
    #         output=node_1.outputs[0],
    #     )
    #     node_2 = material.node_tree.nodes.new(
    #         type="ShaderNodeTexCoord",
    #     )
    #     node_2.location = (-700.0, 100.0)
    #     link_3 = material.node_tree.links.new(
    #         input=node_1.inputs[0],
    #         output=node_2.outputs[2],
    #     )
    #     node_1.inputs["Scale"].default_value = (0.5, 1.0, 1.0)

def register():
    """
    called upon enabling
    """
    # global _icons
    # _icons = bpy.utils.previews.new()
    # bpy.types.Scene.sna_camerasetup = bpy.props.BoolProperty(
    #     name="CameraSetup", description="", default=False
    # )
    # bpy.utils.register_class(PS_Preferences)
    bpy.types.TOPBAR_MT_file_import.append(add_to_topbar)
    bpy.utils.register_class(PS_LoadImage)
    bpy.utils.register_class(PS_LoadDepth)


def unregister():
    """
    called upon disabling
    """
    # global _icons
    # bpy.utils.previews.remove(_icons)
    # wm = bpy.context.window_manager
    # kc = wm.keyconfigs.addon
    # for km, kmi in addon_keymaps.values():
    #     km.keymap_items.remove(kmi)
    # addon_keymaps.clear()
    # del bpy.types.Scene.sna_camerasetup
    # bpy.utils.unregister_class(PS_Preferences)
    bpy.types.TOPBAR_MT_file_import.remove(add_to_topbar)
    bpy.utils.unregister_class(PS_LoadImage)
    bpy.utils.unregister_class(PS_LoadDepth)

if __name__ == "__main__":
    register()