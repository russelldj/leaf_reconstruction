import numpy as np
from copy import copy

# Generated using the MATLAB camera calibration app
#          FocalLength: [8.0742e+03 8.0810e+03]
#       PrincipalPoint: [1.2867e+03 1.1241e+03]
#            ImageSize: [2056 2454]
#     RadialDistortion: [0.2160 -2.5620]
# TangentialDistortion: [0 0]
#                 Skew: 0
#      IntrinsicMatrix: [3×3 double]
MATLAB_RADIAL_DISTORTION: np.array([0.2160 - 2.5620])
MATLAB_K = np.array([[8074.2, 0, 1286.7], [0, 8081.0, 1124.1], [0, 0, 1.0]])
ARTIFICIALLY_CENTERED_K = copy(MATLAB_K)
ARTIFICIALLY_CENTERED_K[0, 2] = 1321

