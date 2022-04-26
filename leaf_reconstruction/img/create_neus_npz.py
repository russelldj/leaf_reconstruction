import os
import argparse
from venv import create

from black import out
import numpy as np
from leaf_reconstruction.config import ARTIFICIALLY_CENTERED_K

from leaf_reconstruction.extrinsics.get_extrinsics import create_projection_matrices

intrinsics_file = "intrinsics.npy"
scale_constant = 1 / 5
OUTPUT_FILE = "cameras_sphere.npz"


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    return args


def save_npz_from_np_arrays(degrees, output_dir, intrinsics=ARTIFICIALLY_CENTERED_K):
    npz_dict = {}
    homogs, _ = create_projection_matrices(None, degrees=degrees)
    for homog, degree in zip(homogs, degrees):

        world_mat = np.eye(4)
        world_mat[0:3, :] = intrinsics @ homog

        scale_mat = np.eye(4)
        scale_mat[0:3, 0:3] = np.eye(3) * scale_constant

        world_mat_key = f"world_mat_{degree:03d}"
        scale_mat_key = f"scale_mat_{degree:03d}"

        npz_dict[world_mat_key] = world_mat
        npz_dict[scale_mat_key] = scale_mat
    outfile = os.path.join(output_dir, OUTPUT_FILE)
    np.savez(outfile, **npz_dict)


def create_npz(calibration_dir, output_dir):
    npz_dict = {}

    intrinsics_path = os.path.join(calibration_dir, intrinsics_file)
    if not os.path.exists(intrinsics_path):
        raise RuntimeError("No instrinsics file found")

    intrinsics = np.load(intrinsics_path)

    for filename in os.listdir(calibration_dir):
        if not filename.endswith(".npy"):
            continue

        if not filename.startswith("extrinsics"):
            continue

        suffix = filename.split("_")[1].split(".npy")[0]

        extrinsics_path = os.path.join(calibration_dir, filename)
        extrinsics = np.load(os.path.join(extrinsics_path))

        world_mat = np.eye(4)
        world_mat[0:3, :] = intrinsics @ extrinsics

        scale_mat = np.eye(4)
        scale_mat[0:3, 0:3] = np.eye(3) * scale_constant

        world_mat_key = "world_mat_" + suffix
        scale_mat_key = "scale_mat_" + suffix

        npz_dict[world_mat_key] = world_mat
        npz_dict[scale_mat_key] = scale_mat

    outfile = os.path.join(output_dir, OUTPUT_FILE)
    np.savez(outfile, **npz_dict)


if __name__ == "__main__":

    args = parse_args()

    calibration_dir = args.calibration_dir
    output_dir = args.output_dir

    if not os.path.exists(calibration_dir):
        raise RuntimeError("Invalid calibration image dir")

    if not os.path.exists(output_dir):
        raise RuntimeError("Invalid output dir")

    create_npz(calibration_dir, output_dir)

