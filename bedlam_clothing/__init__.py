# #  Copyright (c) 2025 Max Planck Society
# #  License: https://bedlam2.is.tuebingen.mpg.de/license.html

from __future__ import annotations

from pathlib import Path

from .config import *

__all__ = ['init_cfg', 'get_cfg', 'get_fnames']


def get_fnames(path: Path | str):
    path = Path(path)

    if path.is_dir():
        fnames = path.rglob('**/*.npz')
    else:
        fnames = [path]

    return fnames
