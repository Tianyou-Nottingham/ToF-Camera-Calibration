"""
This file is for ToF data plane fitting using RANSAC algorithm.
1. Read the data from the serial port or file.
2. Choose the resolution of the data. (Because the raw data is in 8x8 zones, but there must be a point in the zone of which value 
    is the zone's value. For example, if the zones is 8x8, we choose a refine resolution of 256*256. It means each zone has 32*32
    points. And there is a point's depth is just equal to the zone's value.)
3. For every zones, random chose a point. And randomly choose at least 3 zones to fit a plane.
4. Don't take other points in the same zone into account. Just take the chosed one point in the zone to apply RANSAC.
5. RANSAC algorithm
6. Choose other points in the same zone to apply RANSAC.
"""

from read_data_utils import read_serial_data
import numpy as np
import cv2
import time
import serial
import configs.config as cfg
from direction_visualization import refine_by_time
from read_data_utils import visualize2D, normalize
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utils.distance_rectified_fov import distance_rectified_fov


class Plane:
    def __init__(self, N, d) -> None:
        self.N = N
        self.d = d
        self.error = 0

    def solve_distance(self, point):
        """
        计算点到平面的距离
        平面方程: N * x + d = 0
        """
        distance = np.abs(np.dot(self.N, point) - self.d) ** 2
        return distance

    def solve_plane(self, A, B, C):
        """
        求解平面方程
        :params: three points
        :return: Nx(平面法向量), d
        """
        self.N = np.cross(B - A, C - A)
        self.N = self.N / np.linalg.norm(self.N)
        self.d = np.dot(self.N, A + B + C) / 3
        # self.N = self.N if self.N[2] > 0 else -self.N
        # self.d = self.d if self.N[2] > 0 else -self.d

    def fit_plane(self, pts, initial_est=[0, 0, 1, 0.5]):
        """
        Fit a plane given by ax+d = 0 to a set of points
        Works by minimizing the sum over all points x of ax+d
        Ars:
            pts: array of points in 3D space
        Returns:
            (3x1 numpy array): a vector for plane equation
            (float): d in plane equation
            (float): sum of residuals for points to plane (orthogonal l2 distance)
        """

        pts = np.array(pts)

        def loss_fn(x, points):
            self.N = np.array(x[:3])
            self.d = x[3]

            loss = 0
            for point in points:
                loss += (np.dot(self.N, np.array(point)) - self.d) ** 2

            return loss

        def a_constraint(x):
            return np.linalg.norm(x[:3]) - 1

        soln = minimize(
            loss_fn,
            np.array(initial_est),
            args=(pts),
            method="slsqp",
            constraints=[{"type": "eq", "fun": a_constraint}],
            bounds=[(-1, 1), (-1, 1), (-1, 1), (0, None)],
        )

        self.N = soln.x[:3]
        self.d = soln.x[3]
        self.error = np.sqrt(soln.fun / len(pts))

        return self

    def plane_visualization(self, fig, N, d, data, color="r"):
        """
        平面可视化
        :param N: 平面的法向量(3维,形如 [a, b, c])
        :param d: 平面基点偏移量(3维向量,形如 [d_x, d_y, d_z])
        平面方程: N · X + d = 0 或 ax + by + cz + d_offset = 0
        """
        # 解析法向量
        a, b, c = N

        # 创建网格 (x, y)
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)

        # 根据平面方程 N · X + d = 0 求解 z
        if c == 0:
            raise ValueError(
                "The normal vector's z component (c) cannot be zero for visualization."
            )
        Z = -(a * X + b * Y - d) / c

        # 创建 3D 图形
        ax = fig.add_subplot(121, projection="3d")

        # 绘制平面
        # if Z < 0:
        #     ax.plot_surface(-X, -Y, -Z, alpha=0.5, color='blue', edgecolor='k')
        # else:
        ax.plot_surface(X, Y, Z, alpha=0.5, color="blue", edgecolor="k")

        # 绘制数据点
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, marker="o")
        ax.view_init(elev=130, azim=-90, roll=-90)
        # 设置轴标签
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")

        # 设置标题
        ax.set_title("Plane Visualization")

        # 显示图形
        # plt.show()

    def ToF_visualization(self, fig, distance, sigma, res=8, expansion_res=256):
        depth, sigma = visualize2D(distance, sigma, res, expansion_res)
        ax = fig.add_subplot(122)
        norm = plt.Normalize(0, 300)
        ax.imshow(depth, cmap="magma")
        ax.axis("off")
        ax.set_title("ToF Depth Image")
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"), ax=ax, shrink=0.8)
        cb.set_label("Depth Value (mm)")
        
        # ax = fig.add_subplot(122)
        # ax.imshow(color_sigma)
        # ax.set_title('Sigma Image')
        # plt.show()

    def ToF_RANSAC(self, data, res=8, expansion_res=32):
        """
        ToF数据平面拟合
        :params: data: obstacl or safe 3D points
        :params: res: resolution of the ToF data
        :params: expansion_res: resolution of the expansion data
        :return: best_plane: 最优平面
        """
        if len(data) < 3:
            raise ValueError("The number of points should be more than 3.")

        pad_size = expansion_res // res
        [x_index, y_index, d_value] = [data[:, 0], data[:, 1], data[:, 2]]

        best_plane = None
        best_error = np.inf
        iters = 1000
        sigma = 3  ## 阈值
        pre_inlier = 0  ## 内点个数
        Per = 0.99  ## 正确概率
        k = 0
        while (
            k < iters  # and best_error > max_error
        ):  # pretotal < len(data) *2/3: ## 当内点个数大于总点数的3/4 或 大于预设迭代次数时，停止迭代
            ## 随机选择5个点, 拟合平面
            pts_index = data[np.random.choice(len(data), 5, replace=False)]
            pts = []
            for pt_index in pts_index:
                pts.append(
                    np.array(
                        [
                            pt_index[0] + np.random.choice(pad_size, 1)[0] / pad_size,
                            pt_index[1] + np.random.choice(pad_size, 1)[0] / pad_size,
                            pt_index[2],
                        ]
                    )
                )
            # self.solve_plane(pts[0], pts[1], pts[2])
            self.fit_plane(pts)
            total_inlier = 0
            error = 0
            ## 每个zones只取一个点进行RANSAC
            point_offset = [
                np.random.choice(pad_size, len(data)),
                np.random.choice(pad_size, len(data)),
            ]
            for i in range(len(data)):
                point = np.array(
                    [
                        x_index[i] + point_offset[0][i] / pad_size,
                        y_index[i] + point_offset[1][i] / pad_size,
                        d_value[i],
                    ]
                )
                # if cfg.Code["distance_rectified_fov"]:
                #     point = distance_rectified_fov(np.array([point]))[0]
                point_res = self.solve_distance(point)
                if point_res < sigma:
                    total_inlier += 1
                    error += point_res**2
            self.error = np.sqrt(error / total_inlier) if total_inlier > 0 else np.inf
            if total_inlier > pre_inlier:
                iters = np.log(1 - Per) / np.log(1 - pow(total_inlier / len(data), 5))
                pre_inlier = total_inlier
            if self.error < best_error and self.error > 0:
                best_error = self.error
                best_N = self.N
                best_d = self.d
                best_plane = Plane(best_N, best_d)
                best_plane.error = best_error
            ## 当内点个数大于总点数的3/4时，停止迭代
            if total_inlier > len(data) * 3 / 4:
                break

            k += 1
            # print(
            #     f"iter: {k}, total_inlier: {total_inlier}, plane: {self.N, self.d}, error: {self.error},  best_error: {best_error}"
            # )
        return best_plane


def test():
    ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
    last_distances = np.zeros((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    last_sigma = np.ones((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    points3D = []
    while True:
        ## 1. Read the data from the serial port
        distances, sigma = read_serial_data(ser, cfg.Sensor["resolution"])

        ## 2. Refine by time
        time_refine_distances, time_refine_sigma = refine_by_time(
            distances, sigma, last_distances, last_sigma
        )
        points3D = np.array(
            [
                [i, j, time_refine_distances[i, j]]
                for i in range(cfg.Sensor["resolution"])
                for j in range(cfg.Sensor["resolution"])
            ]
        )
        last_distances = distances
        last_sigma = sigma
        ## 2.1 Rectified the distance
        if cfg.Code["distance_rectified_fov"]:
            points_world = distance_rectified_fov(points3D)
        ## 3. ToF RANSAC
        plane = Plane(np.array([1, 0.5, 1]), 50)
        best_plane = plane.ToF_RANSAC(points_world, cfg.Sensor["resolution"], 256)
        # best_plane = plane.fit_plane(points_world)
        print(f"Plane N: {best_plane.N}, d: {best_plane.d}. Error: {best_plane.error}")
        ## 4. Visualization
        depth, sigma = visualize2D(
            time_refine_distances,
            sigma,
            cfg.Sensor["resolution"],
            cfg.Sensor["output_shape"],
        )

        color_depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
        fig = plt.figure(figsize=(14, 7))
        plane.plane_visualization(fig, plane.N, plane.d, points_world)
        plane.ToF_visualization(
            fig,
            time_refine_distances,
            sigma,
            cfg.Sensor["resolution"],
            cfg.Sensor["output_shape"],
        )
        plt.show()
        plt.close(fig)
        # cv2.imshow('depth', color_depth)
        # cv2.waitKey(1) & 0xFF == ord('q')



if __name__ == "__main__":
    test()