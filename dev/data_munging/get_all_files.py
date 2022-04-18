import argparse
from pathlib import Path

from leaf_reconstruction.files.utils import (
    ensure_dir_normal_bits,
    get_files,
    pad_filename,
)
from numpy import pad
from ubelt import ensuredir, symlink
from dev.image_processing.space_carving_example import main as spacecarving_main
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()
    return args


def get_files_in_folder(input_dir):
    files = get_files(input_dir, "**/*.png", require_file=True)
    file_numbers = [pad_filename(f.parts[-2], 7)[0] for f in files]
    # extensions = [f.suffix for f in files]
    int_numbers = [int(x) for x in file_numbers]
    return files, int_numbers


def main(input_dir):
    folders = get_files(input_dir, "*", require_dir=True)
    # Exclude the one taken from the top
    for folder in tqdm(folders):
        print(f"Processing {folders}")
        files, degrees = get_files_in_folder(folder)
        spacecarving_main(files, degrees=degrees)


if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir)
