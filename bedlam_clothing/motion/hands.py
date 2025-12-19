import numpy as np


def swap_hand_poses(poses):
    # swaps the hand pose parameters and negates the second and third dimension of the axis-angle representation
    old_left_hand_pose = poses[:, 75:120].copy().reshape(poses.shape[0], -1, 3)
    old_right_hand_pose = poses[:, 120:165].copy().reshape(poses.shape[0], -1, 3)
    new_left_hand_pose = old_right_hand_pose * np.array([[[0, -1, -1]]])
    new_right_hand_pose = old_left_hand_pose * np.array([[[0, -1, -1]]])
    poses[:, 75:120] = new_left_hand_pose.reshape(poses.shape[0], -1)
    poses[:, 120:165] = new_right_hand_pose.reshape(poses.shape[0], -1)
    return poses
