#  Copyright (c) 2025 Max Planck Society
#  License: https://bedlam2.is.tuebingen.mpg.de/license.html
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf

MOTION_KEYS_TO_KEEP = ['gender', 'surface_model_type', 'mocap_frame_rate', 'mocap_time_length', 'trans', 'poses',
                       'betas']
MOTION_FRAME_DEPENDENT_PARAMS = ['poses', 'trans', 'root_orient']

__all__ = ['get_cfg', 'init_cfg']

_cfg: DictConfig | None = None


def init_cfg(cfg_path: str | Path, dataset_cfg_fname: str | Path = None):
    cfg_path = Path(cfg_path)
    dataset_cfg_fname = Path(dataset_cfg_fname) if dataset_cfg_fname is not None else None

    # replace=True keeps init_cfg callable more than once in the same process.
    OmegaConf.register_new_resolver("to_upper", lambda s: s.upper(), replace=True)

    cfgs = [
        OmegaConf.load(cfg_path / 'motion_processing.yaml'),
        OmegaConf.load(cfg_path / 'body_models.yaml'),
        OmegaConf.load(cfg_path / 'pose_normalization_matrices.yaml')
    ]

    if dataset_cfg_fname is not None:
        cfgs += [OmegaConf.load(dataset_cfg_fname)]

    global _cfg
    _cfg = OmegaConf.merge(*cfgs)

    _cfg.keys_to_keep = MOTION_KEYS_TO_KEEP
    _cfg.frame_dependent_params = MOTION_FRAME_DEPENDENT_PARAMS

    return _cfg


def get_cfg() -> DictConfig:
    global _cfg
    if _cfg is None:
        raise ValueError("Configuration has not been initialized. Please call 'init_cfg' first.")
    return _cfg
