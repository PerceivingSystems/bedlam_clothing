import os
from pathlib import Path
from typing import Union

import bpy

__all__ = ['import_alembic', 'save_scene', 'import_mesh_with_cache']


def import_alembic(abc_fname: Union[str, Path]):
    bpy.ops.wm.alembic_import(filepath=str(abc_fname),
                              # set_frame_range=True,
                              relative_path=True, as_background_job=False)


def save_scene(out_blend_fname, self_contained=True, **kwargs):
    if self_contained:
        bpy.ops.file.pack_all()
    try:
        os.remove(out_blend_fname)
    except FileNotFoundError:
        pass
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend_fname), **kwargs)


def import_mesh_with_cache(base_mesh_obj_fname, pc2_fname, frame_start=0):
    bpy.ops.wm.obj_import(filepath=base_mesh_obj_fname)
    bpy.ops.object.modifier_add(type='MESH_CACHE')
    mesh = bpy.context.object
    mesh.modifiers["MeshCache"].cache_format = 'PC2'
    mesh.modifiers["MeshCache"].filepath = pc2_fname
    mesh.modifiers["MeshCache"].frame_start = frame_start
    mesh.rotation_euler[0] = 0
    mesh.data.materials.clear()
    return mesh
