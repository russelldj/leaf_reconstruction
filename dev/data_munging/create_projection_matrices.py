from pathlib import Path
from venv import create
from leaf_reconstruction.extrinsics.get_extrinsics import create_projection_matrices


def convert_screen_to_NDC(
    image_width, image_height, fx_screen, fy_screen, px_screen, py_screen
):
    s = min(image_width, image_height)
    fx_ndc = fx_screen * 2.0 / s
    fy_ndc = fy_screen * 2.0 / s

    px_ndc = -(px_screen - image_width / 2.0) * 2.0 / s
    py_ndc = -(py_screen - image_height / 2.0) * 2.0 / s
    return fx_ndc, fy_ndc, px_ndc, py_ndc


if __name__ == "__main__":
    folder = Path("data/sample_images")
    files = list(sorted(folder.glob("*")))
    files = [f for f in files if "seg" not in str(f)]

    image_size = (2454, 2056)

    A, b = create_projection_matrices(files)
    breakpoint()

