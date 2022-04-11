from pathlib import Path
from venv import create

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from leaf_reconstruction.config import MATLAB_K
from leaf_reconstruction.spatial.space_carving import space_carving
from pyvista import PolyData
from scipy.spatial.transform import Rotation
from skimage.color import rgb2hsv
from skimage.measure import label


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


def create_projection_matrices(files, dist=2.5, K=MATLAB_K):
    C = np.expand_dims(np.array((dist, 0, 0)), axis=1)
    degrees = [int(x.stem) for x in files]
    rotation_matrices = [
        Rotation.from_euler("xyz", (90, angle, 0), degrees=True).as_matrix()
        for angle in degrees
    ]
    print(rotation_matrices)
    # ts = [np.expand_dims(-R @ C, axis=1) for R in rotation_matrices]
    homogs = [np.concatenate((R, C), axis=1) for R in rotation_matrices]
    Ps = [K @ homog for homog in homogs]
    return Ps


hue = np.array([0.051, 0.503])
saturation = np.array([0.102, 0.804])
value = np.array([0.000, 0.786])

lower_bounds, upper_bounds = zip(hue, saturation, value)

FOLDER = Path(
    "/home/frc-ag-1/data/learning_3D_plants/10sides_transformed/88-181-Maize01/2017-09-09/all_imgs"
)
files = list(FOLDER.glob("*"))
projections = create_projection_matrices(files)

segmentations = [segment(file, lower_bounds, upper_bounds) for file in files]

good_points = space_carving(projections, segmentations, threshold=1)
pc = PolyData(good_points[:, :3])
pc.plot()
breakpoint()
