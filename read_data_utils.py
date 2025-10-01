import cv2
import numpy as np
import serial
import matplotlib
import matplotlib.pyplot as plt
import time
import os

import h5py

matplotlib.use("TkAgg", force=True)
from scipy import interpolate
import configs.config as cfg
from utils.distance_rectified_fov import distance_rectified_fov


# PythonDataStart
# |    63  :      5 |    66  :      5 |    71  :      5 |    73  :      5 |    74  :      5 |    79  :      5 |    83  :      5 |     X  :      X |
# |    66  :      5 |    69  :      5 |    74  :      5 |    74  :      5 |    76  :      5 |    81  :      5 |     X  :      X |     X  :      X |
# |    70  :      5 |    71  :      5 |    76  :      5 |    76  :      5 |    79  :      5 |    84  :      5 |    92  :      5 |    96  :      5 |
# |    73  :      5 |    74  :      5 |    78  :      5 |    79  :      5 |    85  :      5 |    91  :      5 |    92  :      5 |   100  :      5 |
# |    73  :      5 |    78  :      5 |    81  :      5 |    85  :      5 |    87  :      5 |     X  :      X |    96  :      5 |   105  :      5 |
# |    78  :      5 |    80  :      5 |    86  :      5 |    87  :      5 |    92  :      5 |     X  :      X |   102  :      5 |   100  :      5 |
# |    83  :      5 |    86  :      5 |    91  :      5 |    91  :      5 |     X  :      X |    99  :      5 |   104  :      5 |    95  :      5 |
# |    84  :      5 |    91  :      5 |    96  :      5 |    98  :      5 |   103  :      5 |    98  :     10 |    99  :      5 |    82  :      5 |
def normalize(value, vmin=cfg.Sensor["min_depth"], vmax=cfg.Sensor["max_depth"]):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.0
    value = (value * 255.0).astype(np.uint8)
    return value


def parse_data(raw_data):
    distance = []
    sigma = []
    mask = []

    for line in raw_data:
        row_first = []
        row_second = []
        mask_i = []
        pairs = line.split(b"|")[
            1:-1
        ]  # Split by '|' and ignore the first and last empty strings
        for pair in pairs:
            first, second = pair.split(b":")
            if first.strip() == b"X":
                row_first.append(1)
                row_second.append(0)
                mask_i.append(0)
            else:
                row_first.append(int(first.strip()))
                row_second.append(int(second.strip()))
                mask_i.append(1)
            
        distance.append(row_first)
        sigma.append(row_second)
        mask.append(mask_i)
    distance.reverse()
    sigma.reverse()
    mask.reverse()
    return np.array(distance), np.array(sigma), np.array(mask)


def read_serial_data(ser, res=8):
    if ser.readline().strip() == b"PythonDataStart":
        raw_data = [ser.readline().strip() for _ in range(res)]
    distances, sigma, mask = parse_data(raw_data)
    return distances, sigma, mask ## 8x8 array


def refine_by_time(last_distances, last_sigma, distances, sigma, res=8):
    refine_distance = np.zeros_like(distances)
    refine_sigma = np.zeros_like(sigma)
    for i in range(res):
        for j in range(res):
            if sigma[i][j] == 0:
                refine_distance[i][j] = last_distances[i][j]
                refine_sigma[i][j] = last_sigma[i][j]
                continue
            elif last_sigma[i][j] == 0:
                refine_distance[i][j] = distances[i][j]
                refine_sigma[i][j] = sigma[i][j]
                continue
            else:
                refine_distance[i][j] = (
                    last_distances[i][j] * sigma[i][j]
                    + distances[i][j] * last_sigma[i][j]
                ) / (sigma[i][j] + last_sigma[i][j])
                refine_sigma[i][j] = (sigma[i][j] * last_sigma[i][j]) / (
                    sigma[i][j] + last_sigma[i][j]
                )
    return refine_distance, refine_sigma


def visualize3D(distances, sigma, res=8, output_shape=[640, 640], upsample=False):
    for i in range(distances.shape()[0]):
        for j in range(distances.shape()[1]):
            plt.scatter(i, j, distances[i, j], c="r")
    plt.show()


def visualize2D(distances, sigma, res=8, output_shape=[640, 640], upsample=False):
    depth_map = np.zeros(output_shape).astype(np.uint8)
    sigma_map = np.zeros(output_shape).astype(np.uint8)
    out_width, out_height = output_shape
    pad_size = out_width // res
    distances = distances.astype(np.uint8)
    sigma = sigma.astype(np.uint8)
    # print(distances)

    if upsample:
        depth_map = cv2.resize(
            distances, (out_width, out_height), interpolation=cv2.INTER_CUBIC
        )
        sigma_map = cv2.resize(
            sigma, (out_width, out_height), interpolation=cv2.INTER_CUBIC
        )
    else:
        for i in range(res):
            for j in range(res):
                depth_map[
                    i * pad_size : (i + 1) * pad_size, j * pad_size : (j + 1) * pad_size
                ] = distances[i, j]
                sigma_map[
                    i * pad_size : (i + 1) * pad_size, j * pad_size : (j + 1) * pad_size
                ] = sigma[i, j]
    # depth = cv2.applyColorMap(normalize(depth), cv2.COLORMAP_MAGMA)
    # cv2.imshow('depth', depth)
    # img_name = f'output/{time.time()}.png'
    # data['depth_image'] = depth
    # cv2.imwrite(img_name, depth)
    # cv2.waitKey(1) & 0xFF == ord('q')
    return depth_map, sigma_map


def read_file_data(file_name, res):
    with open(file_name, "r") as f:
        while f.readline().strip() != "PythonDataStart":
            pass
        # read 8 lines of data
        data = [f.readline().strip() for _ in range(res)]
    return data


def save_h5(data, file_name, h5_cfg):
    with h5py.File(file_name, "w") as f:
        if h5_cfg["distance"]:
            f.create_dataset("distance", data=data["distance"])
        if h5_cfg["sigma"]:
            f.create_dataset("sigma", data=data["sigma"])
        if h5_cfg["rgb"]:
            f.create_dataset("rgb", data=data["rgb"])
        if h5_cfg["depth_image"]:
            f.create_dataset("depth_image", data=data["depth_image"])


def save_distance_data(data, file_name):
    with open(file_name, "w") as f:
        for row in data:
            f.write("|".join([str(x) for x in row]) + "\n")


if __name__ == "__main__":
    ######### Define the source of the data! #########
    source = "serial"  # 'file' or 'serial'

    ######### File Configuration #########
    file_name = "CoolTerm Capture 2024-10-02 11-37-02.txt"

    data = {}
    distance_tag = "100"  ## mm
    if source == "file":
        raw_data = read_file_data(file_name, cfg.Sensor["resolution"])

    else:
        ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
        while True:
            distances, sigma, mask = read_serial_data(ser, cfg.Sensor["resolution"])
            points3D = np.array(
                [
                    [i, j, distances[i, j]]
                    for i in range(cfg.Sensor["resolution"])
                    for j in range(cfg.Sensor["resolution"])
                ]
            )
            # # distances = normalize(distances)
            print(distances)
            # if cfg.Code["distance_rectified_fov"]:
            #     points_world = distance_rectified_fov(points3D)
            # gradx = np.gradient(points_world, axis=0)
            # grady = np.gradient(points_world, axis=1)
            # # print(f"gradx: {gradx}")
            # # print(f"grady: {grady}")
            # ax = plt.axes(projection="3d")
            # ax.scatter(
            #     points_world[:, 0], points_world[:, 1], points_world[:, 2], c="r"
            # )
            # ax.scatter(20 * points3D[:, 0], 20 * points3D[:, 1], points3D[:, 2], c="b")
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # ax.yaxis.set_transform(plt.gca().transAxes)
            # ax.set_zlabel("Z")
            # ax.set_xlim(-100, 100)
            # ax.set_ylim(-100, 100)
            # ax.set_zlim(400, 500)
            # plt.show()
            depth_map, sigma_map = visualize2D(
                distances,
                sigma,
                cfg.Sensor["resolution"],
                cfg.Sensor["output_shape"],
                upsample=False,
            )
            data["distance"] = distances
            data["sigma"] = sigma
            data["depth_image"] = depth_map
            color_depth = cv2.applyColorMap(normalize(depth_map), cv2.COLORMAP_MAGMA)
            cv2.imshow("depth", color_depth)
            cv2.waitKey(1) & 0xFF == ord("q")

            ## Is there the target filefolder?
            # if not os.path.exists(f"output/{distance_tag}"):
            #     os.makedirs(f"output/{distance_tag}")
            # h5_name = f"output/{distance_tag}/{time.time()}.h5"
            # save_h5(data, h5_name, cfg.h5_cfg)

        # ser.close()
