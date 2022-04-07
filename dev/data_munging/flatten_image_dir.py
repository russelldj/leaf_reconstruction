import argparse
from pathlib import Path

from leaf_reconstruction.files.utils import (
    ensure_dir_normal_bits,
    get_files,
    pad_filename,
)
from numpy import pad
from ubelt import ensuredir, symlink


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    return args


def main(input_dir, output_dir):
    files = get_files(input_dir, "**/*.png", require_file=True)
    # Exclude the one taken from the top
    file_numbers = [pad_filename(f.parts[-2], 3)[0] for f in files[:-1]]
    extensions = [f.suffix for f in files[:-1]]
    output_files = [
        Path(output_dir, file_number + extension)
        for file_number, extension in zip(file_numbers, extensions)
    ]
    ensure_dir_normal_bits(output_dir)
    [symlink(f, of) for f, of in zip(files, output_files)]
    print(output_files)


if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir, args.output_dir)
