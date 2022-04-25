import os
import argparse
import cv2

intrinsics_file = 'intrinsics.npy'
scale_constant = 5
output_file = 'cameras_sphere.npz'

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_dir', required=True)
    parser.add_argument('--output_dir', required=True)

    args = parser.parse_args()
    return args

def create_seg(seg_dir, output_dir):

    for filename in os.listdir(seg_dir):
        if not filename.endswith('_seg.png'):
            continue

        file_path = os.path.join(seg_dir, filename)
        im = cv2.imread(file_path).copy()
        im = im * 255

        output_path = os.path.join(output_dir, filename.replace('_seg', ''))
        cv2.imwrite(output_path, im)



if __name__ == "__main__":
    
    args = parse_args()

    seg_dir = args.seg_dir
    output_dir = args.output_dir

    if not os.path.exists(seg_dir):
        raise RuntimeError('Invalid seg image dir')
    
    if not os.path.exists(output_dir):
        raise RuntimeError('Invalid output dir')

    create_seg(seg_dir, output_dir)
    