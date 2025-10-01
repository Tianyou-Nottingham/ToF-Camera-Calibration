import numpy as np
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import config as cfg

if cfg.Sensor["resolution"] == 8:
    focal_length = 0.648
    FoV = 45  # degrees
    dx = np.tan(math.radians(FoV / 2)) / 4 * focal_length
    dy = dx
    u_0 = 3.5
    v_0 = u_0
elif cfg.Sensor["resolution"] == 4:
    focal_length = 0.648
    FoV = 45
    dx = np.tan(math.radians(FoV / 2)) / 2 * focal_length
    dy = dx
    u_0 = 1.5
    v_0 = u_0


def distance_rectified_fov(pts):
    """
    params: pts: (u, v, distance)
    return: x, y, z
    """
    rectified_pts = []
    for point in pts:
        u, v, distance = point
        # alpha_u = math.atan(np.abs(u - u_0) * dx / focal_length)
        # alpha_v = math.atan(np.abs(v - v_0) * dy / focal_length)
        alpha_uv = math.atan(
            np.sqrt((u - u_0) ** 2 + (v - v_0) ** 2) * dx / focal_length
        )
        x_ToF = distance * (u - u_0) * np.cos(alpha_uv) * dx / focal_length
        y_ToF = distance * (v - v_0) * np.cos(alpha_uv) * dy / focal_length
        # if u == (0 or 1 or 6 or 7) and v == (0 or 1 or 6 or 7): # corner case 
        #     z_ToF = distance * 0.9404948331338918
        # elif u == (0 or 1 or 6 or 7) or v == (0 or 1 or 6 or 7): # edge case
        #     z_ToF = distance * 0.8916354321171438
        # else:
            
        z_ToF = distance  #* np.cos(alpha_uv)
        rectified_pts.append([x_ToF, y_ToF, z_ToF])
    return np.array(rectified_pts)


def test():
    pts = [(3, 3, 100), (4, 4, 200), (5, 5, 300)]
    print(distance_rectified_fov(pts))


if __name__ == "__main__":
    test()
