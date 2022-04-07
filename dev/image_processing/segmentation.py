import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from skimage.color import rgb2hsv

from skimage.measure import label
from pathlib import Path


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


def segment(filename, lower_bounds, upper_bounds):
    image = imread(filename)[..., :3] / 255.0

    thresholded_image = threshold_image(image, lower_bounds, upper_bounds)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image)
    axs[1].imshow(rgb2hsv(image))
    axs[2].imshow(thresholded_image)
    plt.show()


hue = np.array([0.051, 0.503])
saturation = np.array([0.102, 0.804])
value = np.array([0.000, 0.786])

lower_bounds, upper_bounds = zip(hue, saturation, value)

FOLDER = Path(
    "/home/frc-ag-1/data/learning_3D_plants/10sides_transformed/88-181-Maize01/2017-09-09/all_imgs"
)
files = FOLDER.glob("*")
[segment(file, lower_bounds, upper_bounds) for file in files]

