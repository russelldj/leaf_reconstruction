from leaf_reconstruction.files.utils import get_files
from imageio import imread, imwrite
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
import numpy as np

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, " ", y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + "," + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow("image", img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, " ", y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(
            img, str(b) + "," + str(g) + "," + str(r), (x, y), font, 1, (255, 255, 0), 2
        )
        cv2.imshow("image", img)


mins = [576, 578, 579, 579, 579, 577, 576, 575, 574, 574]

maxes = [744, 744, 746, 747, 747, 746, 743, 743, 741, 743]

min_value = np.min(mins)
max_value = np.max(maxes)
center = min_value + max_value  # these were devided by 2 originally
print(center)

FOLDER = "data/sample_images"
files = get_files(FOLDER, "*")
files = [f for f in files if "seg" not in str(f)]
images = [imread(f) for f in files]
width = images[0].shape[1]
# center = int(width / 2) + 85
for img in images:
    img[:, center - 4 : center + 4, :3] = 0
    img = resize(img, (img.shape[0] // 2, img.shape[1] // 2), anti_aliasing=True)
    # cv2.imshow("image", img)
    # cv2.setMouseCallback("image", click_event)
    # cv2.waitKey(0)
    # plt.imshow(img)
    # plt.show()
