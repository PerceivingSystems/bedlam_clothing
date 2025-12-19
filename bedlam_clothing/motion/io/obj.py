#  Copyright (c) 2025 Max Planck Society
#  License: https://bedlam2.is.tuebingen.mpg.de/license.html
#
# Basic functions for reading and writing Wavefront obj files. Does NOT keep into account all the possible options of
# the Wavefront obj specification.

from pathlib import Path
from typing import Optional, Dict
from typing import Union

import numpy as np


def save_mesh_obj(fname: Union[str, Path],
                  v: np.ndarray,
                  f: np.ndarray,
                  uv: Optional[np.ndarray] = None,
                  uvf: Optional[np.ndarray] = None):
    f += 1 if f.min() == 0 else 0

    v_out, f_out = v.astype(str), f.astype(str)

    # np.c_ adds a column (https://numpy.org/doc/stable/reference/generated/numpy.c_.html)
    v_out = np.c_[['v'] * v_out.shape[0], v_out].tolist()

    if uv is not None:
        vt_out = uv.astype(str)
        vt_out = np.c_[['vt'] * vt_out.shape[0], vt_out].tolist()
    else:
        vt_out = []

    if uvf is not None:
        uvf += 1 if uvf.min() == 0 else 0
        f_all_out = np.char.add(np.char.add(f_out, '/'), uvf.astype(str))
        f_all_out = np.c_[['f'] * f_all_out.shape[0], f_all_out].tolist()
    else:
        f_all_out = np.c_[['f'] * f_out.shape[0], f_out].tolist()

    fname = Path(fname).with_suffix('.obj')

    with open(fname, 'w') as f_obj:
        lines = [' '.join(line) for line in (v_out + vt_out + f_all_out)]
        out_data = '\n'.join(lines + [''])
        f_obj.write(out_data)


def read_mesh_obj(fname: Union[str, Path], v_only=False) -> Dict[str, np.ndarray | None]:
    with open(fname, 'r') as f_obj:
        lines = [line for line in f_obj.read().split('\n') if not line.startswith('#') and line != '']

    v, f_all, uv = [], [], []

    for line in lines:
        if line.startswith('v '):
            v.append(line.split(' ')[1:])
        elif v_only:
            continue
        elif line.startswith('vt '):
            uv.append(line.split(' ')[1:])
        elif line.startswith('f '):
            f_all.append([item.split('/') for item in line.split()[1:]])

    v = np.array(v, dtype=np.float32)
    if v_only: return {'v': v, 'uv': None, 'f': None, 'uvf': None}

    uv = np.array(uv, dtype=np.float32) if len(uv) > 0 else None
    f = [[x[0] for x in f_] for f_ in f_all]
    f = np.array(f, dtype=np.int32) - 1
    if len(f_all[0][0]) > 1:
        uvf = [[x[1] for x in f_] for f_ in f_all]
        uvf = np.array(uvf, dtype=np.int32) - 1
    else:
        uvf = None

    return {'v': v, 'uv': uv, 'f': f, 'uvf': uvf}
