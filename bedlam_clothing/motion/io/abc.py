#  Copyright (c) 2025 Max Planck Society
#  License: https://bedlam2.is.tuebingen.mpg.de/license.html
from __future__ import annotations

import tempfile
from pathlib import Path

import bpy
import numpy as np
import trimesh
from bomoto.body_models import BodyModel
from loguru import logger

from bedlam_clothing import get_cfg
from bedlam_clothing.blender import clean_scene, import_mesh_with_cache
from bedlam_clothing.motion import MotionSeq
from bedlam_clothing.motion.io import read_mesh_obj, save_pc2
from bedlam_clothing.utils.rotation import rotate_points_around_axis


def export_as_abc_unreal(
        out_abc_fname: str | Path,
        v_seq: np.ndarray,
        faces: np.ndarray,
        target_fps: float
):
    out_abc_fname = Path(out_abc_fname)

    if out_abc_fname.suffix != ".abc":
        raise ValueError(f"Invalid output path: {out_abc_fname} must have .abc suffix")

    v_seq = rotate_points_around_axis(v_seq.astype(np.float32), deg=90, axis=0).astype(np.float32)

    tmp_pc2_file = tempfile.NamedTemporaryFile(suffix='.pc2', delete=False)
    tmp_obj_file = tempfile.NamedTemporaryFile(suffix='.obj', delete=False)
    save_pc2(tmp_pc2_file.name, v_seq)
    trimesh.Trimesh(vertices=v_seq[0], faces=faces, process=False).export(tmp_obj_file.name)
    tmp_pc2_file.close()
    tmp_obj_file.close()

    clean_scene()
    bpy.context.scene.render.fps = target_fps

    body_mesh = import_mesh_with_cache(tmp_obj_file.name, tmp_pc2_file.name, frame_start=1)
    body_mesh.name = out_abc_fname.stem

    out_abc_fname.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.alembic_export(
        filepath=str(out_abc_fname),  # Specify the path to save your Alembic file
        uvs=True,  # Export UVs
        packuv=True,  # Pack UVs if necessary
        normals=False,  # Export normals
        selected=True,
        vcolors=False,  # Set to True if you want to export vertex colors
        orcos=False,  # Disable original coordinates if not needed
        face_sets=False,  # Disable face sets
        subdiv_schema=False,  # Disable subdivision schema
        apply_subdiv=False,  # Do not apply subdivision
        use_instancing=False,  # Disable instancing for simplicity
        export_hair=False,  # Disable hair export
        export_particles=False,  # Disable particle export
        export_custom_properties=False,  # Disable custom properties export
        # start=...,
        end=v_seq.shape[0]
    )


def motion_npz_to_unreal_abc(
        motion_npz_fname: Path | str,
        out_abc_fname: Path | str,
        use_v_template_in_motion_file: bool = False,
        v_template_fname: Path | str = None,
        zero_betas=False
):
    cfg = get_cfg()

    motion = MotionSeq.load_npz(motion_npz_fname)

    if v_template_fname is not None:
        logger.info(f'Exporting abc cache with provided v_template: {v_template_fname}')
        v_template = read_mesh_obj(v_template_fname, v_only=True)['v']
    elif use_v_template_in_motion_file:
        logger.info(f'Exporting abc cache with provided v_template inside {motion.fname}')
        v_template = motion.v_template
        if v_template is None: raise ValueError('v_template is None')
    else:
        v_template = None

    bmargs = {
        'model_path': cfg.smplx.path,
        'gender': motion.gender,
        'n_betas': motion.n_betas,
        'batch_size': motion.n_frames,
        'device': cfg.device,
        'v_template': v_template
    }

    body_model = BodyModel.instantiate('smplx', **bmargs)

    v_seq = body_model.forward(
        motion.betas if not zero_betas else motion.betas * 0,
        motion.poses,
        motion.trans
    ).detach().cpu().numpy()

    export_as_abc_unreal(
        out_abc_fname=out_abc_fname,
        v_seq=v_seq,
        faces=body_model.faces.detach().cpu().numpy(),
        target_fps=int(cfg.target_fps)
    )
