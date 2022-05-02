import gzip
import json
from collections import defaultdict
from math import radians
from pathlib import Path

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

ROOT_FOLDER = "/home/frc-ag-1/data/co3d/plant"


def write_one_folder(folder, calibration_params, radius):
    scale_mat = (np.eye(3) / radius).tolist()
    output_dict = {"scale": scale_mat, "frame_params": []}
    for capture in calibration_params:
        viewpoint = capture["viewpoint"]
        image = capture["image"]
        path = image["path"]
        path = Path(*Path(path).parts[2:])
        im_size = image["size"]
        viewpoint["path"] = path
        viewpoint["size"] = im_size
        output_dict["frame_params"].append(viewpoint)

    output_file = Path(folder, "params.npz")
    np.savez(output_file, output_dict)


def process_CO3D_data(
    frame_annotation_file,
    quantile=0.95,
    scale_factor=1.2,
    vis=False,
    output_file_pattern="vis/co3d_sphere/{}.png",
):
    with gzip.open(frame_annotations_file, "rb") as infile:
        frame_annotations = json.load(infile)

    by_sequence_dict = defaultdict(list)
    for frame in frame_annotations:
        sequence_name = frame["sequence_name"]
        by_sequence_dict[sequence_name].append(frame)

    for k, v in by_sequence_dict.items():
        folder_name = Path(ROOT_FOLDER, k)
        try:
            pointcloud_name = Path(ROOT_FOLDER, k, "pointcloud.ply")
        except FileNotFoundError:
            continue
        cloud = pv.read(pointcloud_name)
        dists = np.linalg.norm(cloud.points, axis=1)
        kth_quantile = np.quantile(dists, quantile)
        radius = kth_quantile * scale_factor
        if vis:
            # plt.hist(
            #    dists, bins=40, cumulative=True, label="CDF DATA",
            # )
            # plt.show()

            sphere = pv.Sphere(radius=radius)
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(sphere, opacity=0.3)
            plotter.add_mesh(cloud, rgb=True)
            output_file = output_file_pattern.format(k)
            print(output_file)
            plotter.show(screenshot=output_file)
            plotter.close()
            del sphere
            del cloud
            del plotter
        print(pointcloud_name, radius)
        write_one_folder(folder_name, v, radius)


frame_annotations_file = Path(ROOT_FOLDER, "frame_annotations.jgz")
process_CO3D_data(frame_annotations_file, vis=False)
