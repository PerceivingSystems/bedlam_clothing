#  Copyright (c) 2025 Max Planck Society
#  License: https://bedlam2.is.tuebingen.mpg.de/license.html
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from torch.cuda import is_available

from bedlam_clothing import init_cfg, get_cfg, get_fnames
from bedlam_clothing.motion.io import motion_npz_to_unreal_abc

DEVICE = 'cuda' if is_available() else 'cpu'


def export_motions_as_unreal_abc(
        motion_npz_fnames: List[Path | str],
        out_abc_fnames: List[Path | str],
        use_v_template_in_motion_file: bool = False,
        v_template_fname: Path | str = None,
        zero_betas=False
):
    for motion_npz_fname, out_abc_fname in zip(motion_npz_fnames, out_abc_fnames):
        motion_npz_to_unreal_abc(
            motion_npz_fname=motion_npz_fname,
            out_abc_fname=out_abc_fname,
            use_v_template_in_motion_file=use_v_template_in_motion_file,
            v_template_fname=v_template_fname,
            zero_betas=zero_betas
        )


def main():
    parser = argparse.ArgumentParser(
        description='Export animations (from .npz motion files) to Alembic (.abc) for use in Unreal Engine'
    )
    parser.add_argument(
        '--cfg_path',
        type=Path,
        help='Path to base configuration directory (defaults to `config/base_cfg`)',
        default=Path('config/base_cfg'),
        required=False
    )
    parser.add_argument(
        '--motions_npz_path',
        type=Path,
        help='Path to a single .npz motion file or a directory containing .npz motion files (required)',
        required=True
    )
    parser.add_argument(
        '--out_abc_path',
        type=Path,
        help='Directory where exported .abc files will be written (required). Filenames mirror input .npz structure with `.abc` suffix.',
        required=True
    )
    parser.add_argument(
        '--use_v_template_in_motion_file',
        action=argparse.BooleanOptionalAction,
        help='If set, use the vertex template embedded inside each motion .npz when exporting. If not set, an external template can be provided with `--v_template_fname`.',
        default=False
    )
    parser.add_argument(
        '--v_template_fname',
        type=Path,
        help='Path to an external vertex template file (e.g. .obj or .ply). Used when not using the embedded template in the motion file.',
        default=None
    )
    parser.add_argument(
        '--zero_betas',
        action=argparse.BooleanOptionalAction,
        help='If set, zero SMPL shape betas.',
        default=False
    )
    args = parser.parse_args()

    init_cfg(cfg_path=args.cfg_path)
    cfg = get_cfg()
    cfg.device = 'cuda' if is_available() else 'cpu'

    motion_npz_fnames = list(get_fnames(args.motions_npz_path))

    out_abc_fnames = [args.out_abc_path / fname.relative_to(args.motions_npz_path.parent).with_suffix('.abc')
                      for fname in motion_npz_fnames]

    export_motions_as_unreal_abc(
        motion_npz_fnames=motion_npz_fnames,
        out_abc_fnames=out_abc_fnames,
        use_v_template_in_motion_file=args.use_v_template_in_motion_file,
        v_template_fname=args.v_template_fname,
        zero_betas=args.zero_betas
    )


if __name__ == '__main__':
    main()
