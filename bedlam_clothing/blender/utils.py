import bpy

__all__ = ['clean_scene', 'parent_to_vertex']


def clean_scene():
    for o in bpy.context.scene.objects:
        o.select_set(True)
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def parent_to_vertex(obj_to_parent, obj_parent, vertex_idx):
    obj_to_parent.parent = obj_parent
    obj_to_parent.parent_type = 'VERTEX'
    obj_to_parent.parent_vertices = [vertex_idx] * 3
