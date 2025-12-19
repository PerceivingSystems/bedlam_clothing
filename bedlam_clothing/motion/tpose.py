#  Copyright (c) 2025 Max Planck Society
#  License: https://bedlam2.is.tuebingen.mpg.de/license.html

from __future__ import annotations

import numpy as np


def compute_tpose_mask(joints: np.ndarray, return_detailed_info: bool = False,
                       arm_dotprod_threshold: float = 0.8,
                       forearm_dotprod_threshold: float = 0.9,
                       spine_dotprod_threshold: float = 0.9,
                       leg_dotprod_threshold: float = 0.9,
                       facing_dir_dotprod_threshold: float = 0.9) -> np.ndarray:
    """
    Computes a mask that indicates whether the pose is a T-pose or not. The coordinate system is assumed to be the
    default SMPL-X coordinate system (i.e. OpenGL)

    Args:
        joints: np.ndarray
            The joints in the pose. The shape is (n_frames, n_joints, 3)
        return_detailed_info: bool
            Whether to return a dictionary with detailed information about the different criteria used to determine if
            the pose is a T-pose or not. Default is False.
        arm_dotprod_threshold: float
            The threshold for the dot product of the arm vectors to be considered horizontal.
        forearm_dotprod_threshold: float
            The threshold for the dot product of the forearm vectors to be considered horizontal.
        spine_dotprod_threshold: float
            The threshold for the dot product of the spine vector to be considered vertical.
        leg_dotprod_threshold: float
            The threshold for the dot product of the leg vectors to be considered vertical.
        facing_dir_dotprod_threshold: float
            The threshold for the dot product of the facing direction vector to be considered facing the same direction.


    Returns:
        mask: np.ndarray
            A boolean mask indicating whether the pose is a T-pose or not. The shape is (n_frames,)
        flags_dict: dict
            A dictionary containing the different criteria used to determine if the pose is a T-pose or not. The keys
            are the following:
                - 'left_arm_horizontal': Whether the left arm is horizontal
                - 'right_arm_horizontal': Whether the right arm is horizontal
                - 'left_forearm_horizontal': Whether the left forearm is horizontal
                - 'right_forearm_horizontal': Whether the right forearm is horizontal
                - 'spine_vertical': Whether the spine is vertical
                - 'left_leg_vertical': Whether the left leg is vertical
                - 'right_leg_vertical': Whether the right leg is vertical
                - 'facing_dir': Whether the character is facing the same direction as the first frame

    """

    def normalize(v, axis=1):
        return v / np.linalg.norm(v, axis=axis)[:, None]

    def joint_location_dotprod(j1, j2, axis_proj):
        vec = normalize(joints[:, j1, :] - joints[:, j2, :])
        vec_proj = normalize(vec * axis_proj)
        vec_dotprod = (vec * vec_proj).sum(1)
        return vec_dotprod

    left_arm_dotprod = joint_location_dotprod(18, 16, np.array([[1, 0, 1]]))
    right_arm_dotprod = joint_location_dotprod(19, 17, np.array([[1, 0, 1]]))
    left_forearm_dotprod = joint_location_dotprod(20, 18, np.array([[1, 0, 1]]))
    right_forearm_dotprod = joint_location_dotprod(21, 19, np.array([[1, 0, 1]]))
    spine_dotprod = joint_location_dotprod(12, 0, np.array([[0, 1, 0]]))
    left_leg_dotprod = joint_location_dotprod(4, 1, np.array([[0, 1, 0]]))
    right_leg_dotprod = joint_location_dotprod(5, 2, np.array([[0, 1, 0]]))

    cross = lambda a, b: np.cross(a, b)  # a bug in numpy code makes PyCharm autocorrect fail if using np.cross directly

    facing_dir = normalize(cross(joints[:, 2, :] - joints[:, 0, :], joints[:, 1, :] - joints[:, 0, :]))
    facing_dir_dotprod = (facing_dir * facing_dir[0, :]).sum(1)
    flags_dict = {
        'left_arm_horizontal': left_arm_dotprod > arm_dotprod_threshold,
        'right_arm_horizontal': right_arm_dotprod > arm_dotprod_threshold,
        'left_forearm_horizontal': left_forearm_dotprod > forearm_dotprod_threshold,
        'right_forearm_horizontal': right_forearm_dotprod > forearm_dotprod_threshold,
        'spine_vertical': spine_dotprod > spine_dotprod_threshold,
        'left_leg_vertical': left_leg_dotprod > leg_dotprod_threshold,
        'right_leg_vertical': right_leg_dotprod > leg_dotprod_threshold,
        'facing_dir': facing_dir_dotprod > facing_dir_dotprod_threshold,
    }

    all_flags = np.stack(list(flags_dict.values()))
    mask = (~all_flags).sum(axis=0) == 0

    return mask if not return_detailed_info else (mask, flags_dict)


def refine_tpose_mask(mask, fps, tpose_frac_threshold=0.7):
    """
    Refine start/end T-pose labels in a frame-wise boolean mask.

    Args:
        mask (`np.ndarray`): 1-D boolean array of shape (n_frames,).
        fps (int): Frames per second, used to set temporal windows.
        tpose_frac_threshold (float): Fraction of frames within the temporal window that must be labeled as T-pose

    Returns:
        tuple: (refined_mask (`np.ndarray`), (start, end) (int, int))
            start: exclusive end index of the detected start T-pose region (frames `0..start-1` are flagged).
            end: inclusive start index of the detected end T-pose region (frames `end..n_frames-1` are flagged).
    """
    if not 0.0 <= tpose_frac_threshold <= 1.0:
        raise ValueError('tpose_frac_threshold must be in [0.0, 1.0]')

    fps = round(fps)
    start_end_threshold = min(round(2 * fps), mask.shape[0])
    n_unpose_frames = round(fps / 4)

    refined_mask = np.zeros_like(mask, dtype=bool)

    start, end = 0, mask.shape[0]

    for i in range(0, start_end_threshold):
        if mask[i]:
            submask = mask[max(0, i - fps):i + 1]
            tpose_frac = submask.sum() / float(submask.shape[0])
            if tpose_frac > tpose_frac_threshold:
                start = i + n_unpose_frames
                refined_mask[:start] = True

    for i in range(mask.shape[0] - 1, mask.shape[0] - start_end_threshold, -1):
        if mask[i]:
            submask = mask[i:min(mask.shape[0], i + fps)]
            tpose_frac = submask.sum() / float(submask.shape[0])
            if tpose_frac > tpose_frac_threshold:
                end = i - n_unpose_frames
                refined_mask[end:] = True

    return refined_mask, (start, end)


def compute_refined_tpose_mask(joints: np.ndarray, fps: int):
    return refine_tpose_mask(compute_tpose_mask(joints), fps)
