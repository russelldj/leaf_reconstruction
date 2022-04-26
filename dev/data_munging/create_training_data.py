import argparse
from pathlib import Path
from smtpd import DebuggingServer

import matplotlib.pyplot as plt
import numpy as np
from cv2 import imwrite
import cv2
from imageio import imread
from leaf_reconstruction.files.utils import ensure_dir_normal_bits, get_files

# from leaf_reconstruction.img.create_neus_mask import create_seg
from leaf_reconstruction.img.create_neus_npz import save_npz_from_np_arrays
from leaf_reconstruction.img.segment import segment
from skimage.morphology import convex_hull_image, disk, binary_dilation

from get_all_files import get_files_in_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", help="Where the all the capture subfolders are")
    parser.add_argument(
        "--output-dir", help="Where to write all the data in NeuS format"
    )
    args = parser.parse_args()
    return args


def expand_mask(mask, kernel_size=50):
    kernel = disk(kernel_size).astype(bool)
    mask = convex_hull_image(mask)
    # mask = binary_dilation(mask, kernel)
    mask = cv2.dilate(mask.astype(np.uint8), kernel.astype(np.uint8))
    return mask


def create_training_data(input_dir, output_dir, just_npz=True):
    folders = get_files(input_dir, "*", require_dir=True)
    for folder in folders:

        files, degrees = get_files_in_folder(folder)
        files_degs = zip(files, degrees)
        # Sort by degrees
        files_degs = sorted(files_degs, key=lambda x: x[1])
        degrees = [x[1] for x in files_degs]

        output_stem = folder.parts[-1]
        local_output_folder = Path(output_dir, output_stem)
        save_npz_from_np_arrays(degrees, local_output_folder)

        if just_npz:
            continue

        imgs = [imread(file) for file, _ in files_degs]

        segs = [segment(file) for file, _ in files_degs]
        masked_images = []

        for img, seg in zip(imgs, segs):
            not_seg = np.logical_not(seg)
            img[not_seg] = 0
            masked_images.append(img[..., :3])

        output_masks = [expand_mask(mask) for mask in segs]
        output_masks = [(mask * 255).astype(np.uint8) for mask in output_masks]

        local_img_output_folder = Path(local_output_folder, "image")
        local_mask_output_folder = Path(local_output_folder, "mask")
        ensure_dir_normal_bits(local_img_output_folder)
        ensure_dir_normal_bits(local_mask_output_folder)
        output_files = [f"{degree:03d}.png" for _, degree in files_degs]
        output_mask_files = [Path(local_mask_output_folder, f) for f in output_files]
        output_img_files = [Path(local_img_output_folder, f) for f in output_files]
        # Write data
        [
            imwrite(str(f), np.flip(img, axis=2))
            for f, img in zip(output_img_files, masked_images)
        ]
        [imwrite(str(f), seg) for f, seg in zip(output_mask_files, output_masks)]

        plt.imshow(masked_images[0])
        plt.pause(1)


if __name__ == "__main__":
    args = parse_args()
    create_training_data(args.input_dir, args.output_dir)
