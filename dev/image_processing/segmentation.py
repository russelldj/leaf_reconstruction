from pathlib import Path
from venv import create

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread, imwrite
from leaf_reconstruction.config import MATLAB_K
from leaf_reconstruction.spatial.space_carving import space_carving
from pyvista import PolyData
from scipy.spatial.transform import Rotation
from skimage.color import rgb2hsv
from skimage.measure import label
from leaf_reconstruction.vis.vis import visualize
import cv2


def getLargestCC(segmentation):
    """https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image"""
    labels = label(segmentation)
    assert labels.max() != 0  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def threshold_channel(image, lower_bound, upper_bound):
    if lower_bound > upper_bound:
        raise ValueError("Lower bound greater than lower bound")

    thresholded = np.logical_and(image > lower_bound, image < upper_bound)
    return thresholded


def threshold_image(image, lower_bounds, upper_bounds):
    thresholded_channels = [
        threshold_channel(image[..., i], lower_bounds[i], upper_bounds[i])
        for i in range(image.shape[2])
    ]
    mask = np.logical_and.reduce(thresholded_channels)
    mask = getLargestCC(mask)
    return mask


def segment(filename, lower_bounds, upper_bounds, vis=False):
    image = imread(filename)[..., :3] / 255.0

    thresholded_image = threshold_image(image, lower_bounds, upper_bounds)
    if vis:
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(image)
        axs[1].imshow(rgb2hsv(image))
        axs[2].imshow(thresholded_image)
        plt.show()
    return thresholded_image


def create_projection_matrices(files, dist=2.5, K=MATLAB_K, vis=False):
    # The world Z is up and we start at the negative X configuration
    initial_rotation = Rotation.from_euler(
        "xyz", (90, 270, 0), degrees=True
    ).as_matrix()
    C = np.array((0, 0, dist))
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
    return Ps


hue = np.array([0.051, 0.503])
saturation = np.array([0.102, 0.804])
value = np.array([0.000, 0.786])

lower_bounds, upper_bounds = zip(hue, saturation, value)

FOLDER = Path("data/sample_images")
files = list(sorted(FOLDER.glob("*")))
files = [f for f in files if "seg" not in str(f)]
projections = create_projection_matrices(files, vis=False)

segmentations = [segment(file, lower_bounds, upper_bounds) for file in files]
# output_files = [str(x).replace(".png", "_seg.png") for x in files]
# [imwrite(f, i.astype(np.uint8)) for f, i in zip(output_files, segmentations)]
good_points = space_carving(projections, segmentations, threshold=1)
pc = PolyData(good_points[:, :3])
pc.plot()
