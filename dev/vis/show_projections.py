import pyvista as pv
import numpy as np
from argparse import ArgumentParser
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from imageio import imread


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cameras-sphere")
    parser.add_argument("--co3d-output")
    parser.add_argument("--pointcloud")

    args = parser.parse_args()
    return args


# https://github.com/facebookresearch/co3d/blob/7ee9f5ba0b87b22e1dfe92c4d2010cb14dd467a6/dataset/co3d_dataset.py#L490
def thing():
    # principal point and focal length
    principal_point = torch.tensor(entry.viewpoint.principal_point, dtype=torch.float)
    focal_length = torch.tensor(entry.viewpoint.focal_length, dtype=torch.float)

    # first, we convert from the legacy Pytorch3D NDC convention
    # (currently used in CO3D for storing intrinsics) to pixels
    half_image_size_wh_orig = (
        torch.tensor(list(reversed(entry.image.size)), dtype=torch.float) / 2.0
    )

    # principal point and focal length in pixels
    principal_point_px = -1.0 * (principal_point - 1.0) * half_image_size_wh_orig
    focal_length_px = focal_length * half_image_size_wh_orig
    if self.box_crop:
        assert clamp_bbox_xyxy is not None
        principal_point_px -= clamp_bbox_xyxy[:2]

    # now, convert from pixels to Pytorch3D v0.5+ NDC convention
    if self.image_height is None or self.image_width is None:
        out_size = list(reversed(entry.image.size))
    else:
        out_size = [self.image_width, self.image_height]

    half_image_size_output = torch.tensor(out_size, dtype=torch.float) / 2.0
    half_min_image_size_output = half_image_size_output.min()

    # rescaled principal point and focal length in ndc
    principal_point = (
        half_image_size_output - principal_point_px * scale
    ) / half_min_image_size_output
    focal_length = focal_length_px * scale / half_min_image_size_output


def convert_NDC_to_screen(im_w, im_h, fx_ndc, fy_ndc, px_ndc, py_ndc):
    s = min(im_w, im_h)
    px_screen = -(px_ndc * s / 2) + im_w / 2
    py_screen = -(py_ndc * s / 2) + im_h / 2
    fx_screen = fx_ndc * s / 2
    fy_screen = fy_ndc * s / 2
    return fx_screen, fy_screen, px_screen, py_screen


def create_projection(K, R, t, method="naive"):
    if method == "naive":
        extrinsics = np.concatenate((R, np.expand_dims(t, axis=1)), axis=1)
        proj = K @ extrinsics

    return proj


def project_to_image(proj, points):
    homog_image_space = proj @ points
    inhomog_image_space = homog_image_space[:2] / homog_image_space[2:]
    return inhomog_image_space


def extract_K_R_t_from_frame_param(frame_param, K_in_NDC=False):
    R = frame_param["R"]
    t = frame_param["T"]
    principle_point = frame_param["principal_point"]
    focal_length = frame_param["focal_length"]
    K = np.eye(3)
    if K_in_NDC:
        K[0, 0] = focal_length[0]
        K[1, 1] = focal_length[1]
        K[0, 2] = principle_point[0]
        K[1, 2] = principle_point[1]
    else:
        im_h, im_w = frame_param["size"]
        fx_ndc = focal_length[0]
        fy_ndc = focal_length[1]
        px_ndc = principle_point[0]
        py_ndc = principle_point[1]
        fx_screen, fy_screen, px_screen, py_screen = convert_NDC_to_screen(
            im_h=im_h,
            im_w=im_w,
            fx_ndc=fx_ndc,
            fy_ndc=fy_ndc,
            px_ndc=px_ndc,
            py_ndc=py_ndc,
        )
        K[0, 0] = fx_screen
        K[1, 1] = fy_screen
        K[0, 2] = px_screen
        K[1, 2] = py_screen

    return K, R, t


def load_image_from_frame_param(frame_param, frame_param_file):
    image_path = frame_param["path"]
    image_folder = Path(frame_param_file).parents[0]
    image_path = Path(image_folder, image_path)
    image = imread(image_path)
    return image


def downsample_points_colors(points, colors, target_num=10000, no_op=True):
    if no_op:
        return points, colors
    num_points = points.shape[1]
    inds = np.random.choice(num_points, target_num, replace=False)
    # breakpoint()
    downsampled_points = points.T[inds].T
    downsampled_colors = colors[inds]
    return downsampled_points, downsampled_colors


def main(camera_sphere, co3d_file, pointcloud_file):
    co3d_data = np.load(co3d_file, allow_pickle=True)

    co3d_data = co3d_data["arr_0"][None][0]
    frame_params = co3d_data["frame_params"]
    scale = co3d_data["scale"]

    pointcloud = pv.read(pointcloud_file)
    # pointcloud.plot(rgb=True)

    points = pointcloud.points
    colors = pointcloud.active_scalars / 255.0
    homog_points = np.concatenate((points.T, np.ones((1, points.shape[0]))))

    for frame_param in frame_params:
        image = load_image_from_frame_param(frame_param, co3d_file)
        K, R, t = extract_K_R_t_from_frame_param(frame_param)
        proj = create_projection(K, R, t)
        image_space = project_to_image(proj, homog_points)

        fig, axs = plt.subplots(1, 2)

        image_space, downsampled_colors = downsample_points_colors(image_space, colors)
        # print(image_space.shape)
        # plt.scatter(image_space[0], image_space[1])
        axs[0].scatter(image_space[0], image_space[1], c=downsampled_colors)
        axs[0].set_aspect("equal")
        axs[1].imshow(image)
        plt.show()

    for key in co3d_data.keys():
        data = co3d_data[key]

    data = np.load(camera_sphere)
    for key in data.keys():
        print(key)
        print()
        P = data[key]


if __name__ == "__main__":
    args = parse_args()
    main(args.cameras_sphere, args.co3d_output, args.pointcloud)
