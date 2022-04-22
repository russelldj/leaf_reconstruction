import torch

FILE = "/home/frc-ag-1/dev/courses/learning_for_3d_vision/assignment3/data/lego.pth"
data = torch.load(FILE)
cameras = data["cameras"]
print(cameras.keys())
rotations = cameras["R"]
translations = cameras["T"]
focal_points = cameras["focal_length"]
principle_points = cameras["principal_point"]


breakpoint()

