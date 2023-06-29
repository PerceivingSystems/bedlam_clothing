#  Copyright (c) 2023 Max Planck Society
#  License: https://bedlam.is.tuebingen.mpg.de/license.html

from typing import Union

import numpy as np


def rotate_points_around_axis(v: np.ndarray, deg: float, axis: Union[int, str]):
    """
    Takes a set of points and rotates them by the given angle around the given axis.

    Args:
      v: the points to rotate
      deg: the angle to rotate by
      axis: the axis to rotate around. Can be 0 (=x), 1 (=y) or 2 (=z)

    Returns:
      the rotated points.
    """
    rot = np.radians(deg)

    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if isinstance(axis, str):
        axis = axis_map[axis.lower()]

    if axis == 0:
        rot_mat = np.array([[1, 0, 0],
                            [0, np.cos(rot), -np.sin(rot)],
                            [0, np.sin(rot), np.cos(rot)]])
    elif axis == 1:
        rot_mat = np.array([[np.cos(rot), 0, np.sin(rot)],
                            [0, 1, 0],
                            [-np.sin(rot), 0, np.cos(rot)]])
    elif axis == 2:
        rot_mat = np.array([[np.cos(rot), -np.sin(rot), 0],
                            [np.sin(rot), np.cos(rot), 0],
                            [0, 0, 1]])
    else:
        raise ValueError("axis must be 0 (=x), 1 (=y) or 2 (=z)")

    if len(v.shape) == 2:
        return v @ rot_mat.T
    elif len(v.shape) == 3:
        return np.einsum('nvj, ij -> nvi', v, rot_mat)
    else:
        raise ValueError("v must be 2 or 3 dimensional")
