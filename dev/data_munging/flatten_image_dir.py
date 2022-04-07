from pathlib import Path
import argparse

from leaf_reconstruction.files.utils import get_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    return args


def main(input_dir, output_dir):
    files = get_files(input_dir, "*", require_dir=True)
    print(files)


if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir, args.output_dir)
