#  Copyright (c) 2025 Max Planck Society
#  License: https://bedlam2.is.tuebingen.mpg.de/license.html

import struct
from pathlib import Path
from typing import Union, Iterable

import numpy as np

from bedlam_clothing.motion.io import read_mesh_obj


def save_pc2(out_pc2_fname: Union[str, Path], v_seq: np.ndarray):
    # v_seq: (N, V, 3), N is the number of frames, V is the number of vertices
    n_frames, n_verts = v_seq.shape[:2]
    header = struct.pack('<12siiffi', b'POINTCACHE2\0', 1, n_verts, 1, 1, n_frames)
    with open(out_pc2_fname, 'wb') as f:
        f.write(header)
        f.write(v_seq.tobytes())


def load_pc2(pc2_fname: Union[str, Path]):
    with open(pc2_fname, 'rb') as f:
        header = struct.unpack('<12siiffi', f.read(32))
        n_verts = header[2]
        n_frames = header[-1]
        v_seq = np.frombuffer(f.read(), dtype=np.float32).reshape((n_frames, n_verts, 3))
    return v_seq


def objs_to_pc2(fnames: Iterable[Union[str, Path]], out_pc2_fname: Union[str, Path]):
    v_seq = [read_mesh_obj(fname, v_only=True)['v'] for fname in fnames]
    save_pc2(out_pc2_fname, np.stack(v_seq))
