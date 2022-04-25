from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from leaf_reconstruction.config import ARTIFICIALLY_CENTERED_K
from leaf_reconstruction.spatial.space_carving import space_carving
from pyvista import PolyData
from leaf_reconstruction.extrinsics.get_extrinsics import create_projection_matrices
from leaf_reconstruction.img.segment import LOWER_BOUNDS, UPPER_BOUNDS, segment


def main(
    files,
    lower_bounds=LOWER_BOUNDS,
    upper_bounds=UPPER_BOUNDS,
    degrees=None,
    k=ARTIFICIALLY_CENTERED_K,
    pointcloud=False,
):
    extrinsics, projections = create_projection_matrices(
        files, vis=False, degrees=degrees
    )

    segmentations = [segment(file, lower_bounds, upper_bounds) for file in files]
    # output_files = [str(x).replace(".png", "_seg.png") for x in files]
    # [imwrite(f, i.astype(np.uint8)) for f, i in zip(output_files, segmentations)]
    good_points = space_carving(
        extrinsics=extrinsics,
        num_voxels=300,
        K=k,
        silhouettes=segmentations,
        volume_scale=0.4,
        threshold=9,
    )
    if pointcloud:
        pc = PolyData(good_points[:, :3])
        pc.plot()


if __name__ == "__main__":

    folder = Path("data/sample_images")
    files = list(sorted(folder.glob("*")))
    files = [f for f in files if "seg" not in str(f)]
    main(files, LOWER_BOUNDS, UPPER_BOUNDS)
