import numpy as np
import matplotlib.pyplot as plt


def space_carving(
    projections, silhouettes, num_voxels=120, volume_scale=2, threshold=3
):

    x, y, z = np.mgrid[:num_voxels, :num_voxels, :num_voxels]
    pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(float)
    pts = pts.T
    nb_points_init = pts.shape[0]
    xmax, ymax, zmax = np.max(pts, axis=0)
    pts[:, 0] /= xmax
    pts[:, 1] /= ymax
    pts[:, 2] /= zmax
    center = pts.mean(axis=0)
    pts -= center
    pts /= 1
    # pts[:, 2] -= 0.62

    pts = np.vstack((pts.T, np.ones((1, nb_points_init))))

    filled = []
    for P, im in zip(projections, silhouettes):
        img_h, img_w = im.shape[:2]
        uvs = P @ pts
        z_good = uvs[2] > 0
        uvs /= uvs[2, :]
        uvs = np.round(uvs).astype(int)
        x_good = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < img_w)
        y_good = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < img_h)
        good = np.logical_and.reduce((x_good, y_good, z_good))
        indices = np.where(good)[0]
        fill = np.zeros(uvs.shape[1])
        sub_uvs = uvs[:2, indices]
        plt.scatter(sub_uvs[0], sub_uvs[1])
        plt.show()
        res = im[sub_uvs[1, :], sub_uvs[0, :]]
        fill[indices] = res

        filled.append(fill)
    filled = np.vstack(filled)
    occupancy = np.sum(filled, axis=0)

    # Select occupied voxels
    pts = pts.T
    good_points = pts[occupancy > threshold, :]

    return good_points
