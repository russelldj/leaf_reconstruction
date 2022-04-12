import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def space_carving(
    extrinsics, K, silhouettes, num_voxels=120, volume_scale=2, threshold=3, vis=False
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
    pts /= 2.5
    # pts[:, 2] -= 0.62
    plotter = pv.Plotter()
    pts = np.vstack((pts.T, np.ones((1, nb_points_init))))
    if vis:
        pc = pv.PolyData(pts[:3].T)
        for extrinsic in extrinsics:
            print(extrinsic)

        plotter.add_mesh(pc)
        plotter.show()

    filled = []
    for extrinsic, im in zip(extrinsics, silhouettes):
        img_h, img_w = im.shape[:2]
        uvs = K @ extrinsic @ pts
        z_good = uvs[2] > 0
        uvs /= uvs[2, :]
        uvs = np.round(uvs).astype(int)
        x_good = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < img_w)
        y_good = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < img_h)
        good = np.logical_and.reduce((x_good, y_good, z_good))
        indices = np.where(good)[0]
        fill = np.zeros(uvs.shape[1])
        sub_uvs = uvs[:2, indices]
        if vis:
            fig, axs = plt.subplots(1, 2)
            axs[0].scatter(sub_uvs[0], sub_uvs[1])
            axs[1].imshow(im)
            plt.show()
            plt.close()
        res = im[sub_uvs[1, :], sub_uvs[0, :]]
        fill[indices] = res

        filled.append(fill)
    filled = np.vstack(filled)
    occupancy = np.sum(filled, axis=0)

    # Select occupied voxels
    pts = pts.T
    good_points = pts[occupancy > threshold, :]

    return good_points
