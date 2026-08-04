from .obj import read_mesh_obj, save_mesh_obj
from .pc2 import load_pc2, save_pc2, objs_to_pc2

# The abc writers need bpy, which is an optional dependency and pins the Python
# version. Import that module on first use so the obj and pc2 readers stay
# usable without Blender installed.
_ABC_EXPORTS = frozenset({'export_as_abc_unreal', 'motion_npz_to_unreal_abc'})


def __getattr__(name):
    if name in _ABC_EXPORTS:
        from . import abc
        return getattr(abc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = [
    'export_as_abc_unreal',
    'motion_npz_to_unreal_abc',
    'read_mesh_obj',
    'save_mesh_obj',
    'load_pc2',
    'save_pc2',
    'objs_to_pc2'
]
