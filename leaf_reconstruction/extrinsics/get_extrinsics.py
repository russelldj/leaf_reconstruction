import cv2
import numpy as np
from leaf_reconstruction.config import ARTIFICIALLY_CENTERED_K
from leaf_reconstruction.vis.vis import visualize
from scipy.spatial.transform import Rotation


def create_projection_matrices(
    files, dist=2.5, K=ARTIFICIALLY_CENTERED_K, vis=False, degrees=None
):
    """
    Homogenous transformations and P matrices
    """
    # The world Z is up and we start at the negative X configuration
    initial_rotation = Rotation.from_euler(
        "xyz", (90, 270, 0), degrees=True
    ).as_matrix()
    C = np.array((0, 0, dist))

    if degrees is None:
        degrees = [int(x.stem) for x in files]

    rotations = [
        Rotation.from_euler("yxz", (angle, 0, 0), degrees=True) for angle in degrees
    ]
    rotation_matrices = [
        rotation.as_matrix() @ initial_rotation for rotation in rotations
    ]
    ts = [np.expand_dims(C, axis=1) for R in rotation_matrices]

    homogs = [np.concatenate((R, t), axis=1) for R, t in zip(rotation_matrices, ts)]
    Ps = [K @ homog for homog in homogs]

    if vis:
        rotation_rodrigues = [cv2.Rodrigues(R)[0] for R in rotation_matrices]
        visualize(rotation_rodrigues, ts, K)
    return homogs, Ps
