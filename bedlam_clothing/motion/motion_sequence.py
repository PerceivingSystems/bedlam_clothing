#  Copyright (c) 2025 Max Planck Society
#  License: https://bedlam2.is.tuebingen.mpg.de/license.html
from __future__ import annotations

from pathlib import Path

import numpy as np
from bomoto.body_models import BodyModel

from .framerate import convert_framerate
from .stats import compute_ground_coverage
from .. import get_cfg

__all__ = ['MotionSeq']


class MotionSeq:
    def __init__(
            self,
            fname: Path,
            betas: np.ndarray,
            poses: np.ndarray,
            trans: np.ndarray,
            mocap_frame_rate: float,
            gender: str,
            **extra_args
    ):
        self.fname = fname
        self.betas = betas
        self.poses = poses
        self.trans = trans

        self.mocap_frame_rate = mocap_frame_rate
        if isinstance(self.mocap_frame_rate, np.ndarray): self.mocap_frame_rate = self.mocap_frame_rate.item()

        self.gender = gender
        if isinstance(self.gender, np.ndarray): self.gender = self.gender.item()

        self.extra_args = extra_args

    @staticmethod
    def load_npz(motion_fname: Path | str):
        return MotionSeq(fname=Path(motion_fname), **np.load(motion_fname, allow_pickle=True))

    def save_npz(self, out_npz_fname: Path | str):
        np.savez_compressed(
            out_npz_fname,
            betas=self.betas,
            poses=self.poses,
            trans=self.trans,
            mocap_frame_rate=self.mocap_frame_rate,
            gender=self.gender,
            **self.extra_args
        )

    @property
    def n_betas(self):
        return self.betas.shape[-1]

    @property
    def n_frames(self):
        return self.trans.shape[0]

    @property
    def v_template(self):
        return self.extra_args.get('v_template', None)

    def convert_framerate(self, target_fps: float):
        self.poses, self.trans = convert_framerate(self.poses, self.trans, self.mocap_frame_rate, target_fps)
        self.mocap_frame_rate = target_fps
        return self

    def compute_ground_coverage(self):
        return compute_ground_coverage(self.trans)

    def get_vertices(self):
        cfg = get_cfg()

        bmargs = {
            'model_path': cfg.smplx.path,
            'gender': self.gender,
            'n_betas': self.n_betas,
            'batch_size': self.n_frames,
            'device': cfg.device,
            'v_template': self.v_template
        }

        body_model = BodyModel.instantiate('smplx', **bmargs)
        v_seq = body_model.forward(self.betas, self.poses, self.trans).detach().cpu().numpy()
        return v_seq
