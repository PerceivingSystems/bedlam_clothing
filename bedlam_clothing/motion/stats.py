def compute_ground_coverage(trans):
    x = trans[:, 0].max() - trans[:, 0].min()
    z = trans[:, 2].max() - trans[:, 2].min()
    return x, z, x * z
