import numpy as np
import configs.config as cfg
import serial
from TOF_RANSAC import Plane
from read_data_utils import read_serial_data
from direction_visualization import refine_by_time
from obstacle_avoidance import outliers_detection
import matplotlib.pyplot as plt
from utils.kmeans import plane_kmeans
from utils.distance_rectified_fov import distance_rectified_fov
from scipy.optimize import minimize



def two_plane_visualization(fig, Plane1, Plane2, data1, data2):
    """
    两个平面可视化
    :param N1: 平面1的法向量(3维,形如 [a, b, c])
    :param d1: 平面1基点偏移量(3维向量,形如 [d_x, d_y, d_z])
    :param N2: 平面2的法向量(3维,形如 [a, b, c])
    :param d2: 平面2基点偏移量(3维向量,形如 [d_x, d_y, d_z])
    平面方程: N · X + d = 0 或 ax + by + cz + d_offset = 0
    """
    # 解析法向量
    a1, b1, c1 = Plane1.N
    a2, b2, c2 = Plane2.N

    # # 计算偏移量 d_offset
    # d_offset = -(a * d[0] + b * d[1] + c * d[2])

    # 创建网格 (x, y)
    x = np.linspace(-120, 120, 100)
    y = np.linspace(-120, 120, 100)
    X, Y = np.meshgrid(x, y)

    # 根据平面方程 N · X + d = 0 求解 z
    if c1 == 0 or c2 == 0:
        raise ValueError(
            "The normal vector's z component (c) cannot be zero for visualization."
        )
    Z1 = -(a1 * X + b1 * Y - Plane1.d) / c1
    Z2 = -(a2 * X + b2 * Y - Plane2.d) / c2

    # 创建 3D 图形
    ax = fig.add_subplot(121, projection="3d")

    # 绘制平面
    # if Z < 0:
    #     ax.plot_surface(-X, -Y, -Z, alpha=0.5, color='blue', edgecolor='k')
    # else:
    ax.plot_surface(X, Y, Z1, alpha=0.2, color="red")
    # 红色加legend-（Plane1）
    ax.plot_surface(X, Y, Z2, alpha=0.2, color="green")
    plt.legend(["Plane1", "Plane2"])

    # 绘制数据点
    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c="r", marker="o")
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c="g", marker="o")
    ax.view_init(elev=50, azim=0)
    # 设置轴标签
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    # ax.set_xlim(-100, 100)
    # ax.set_ylim(-100, 100)
    # ax.set_zlim(300, 500)

    # 设置标题
    ax.set_title("Plane Visualization")

def two_planes_fitting(points1, points2):
    """
    两个平面拟合
    :params: points1: 平面1的点
    :params: points2: 平面2的点
    :return: best_plane1: 平面1
    :return: best_plane2: 平面2
    :constraints: 平面1和平面2的法向量互相垂直
    """
    plane1 = Plane(np.array([0, 0, 1]), 0)
    plane2 = Plane(np.array([0, 0, 1]), 0)
    def fit_2planes(pts1, pts2, initial_est=[0, 1, 0, 0.5, 1, 1, 1, 0.5]):
        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        pts = np.vstack((pts1, pts2))

        def loss_fn(x, points):
            x1 = x[:4]
            x2 = x[4:]
            plane1.N = np.array(x1[:3])
            plane1.d = x1[3]
            plane2.N = np.array(x2[:3])
            plane2.d = x2[3]
            points1 = points[:len(pts1)]
            points2 = points[len(pts1):]

            loss1 = 0
            loss2 = 0
            for point in points1:
                loss1 += (np.dot(plane1.N, np.array(point)) - plane1.d) ** 2
            for point in points2:
                loss2 += (np.dot(plane2.N, np.array(point)) - plane2.d) ** 2
            return loss1 + loss2
        
        def b_constraint(x):
            return [{'type': 'eq', 'fun': lambda x: np.linalg.norm(x[:3]) - 1},
                    {'type': 'eq', 'fun': lambda x: np.linalg.norm(x[4:7]) - 1},
                    {'type': 'eq', 'fun': lambda x: np.dot(x[:3], x[4:7])}]

        soln = minimize(
            loss_fn,
            np.array(initial_est),
            args=(pts),
            method="slsqp",
            constraints=[{'type': 'eq', 'fun': lambda x: np.linalg.norm(x[:3]) - 1},
                    {'type': 'eq', 'fun': lambda x: np.linalg.norm(x[4:7]) - 1},
                    {'type': 'eq', 'fun': lambda x: np.dot(x[:3], x[4:7])-0}],
            bounds=[(-1, 1), (-1, 1), (-1, 1), (0, None), (-1, 1), (-1, 1), (-1, 1), (0, None)],
        )
        print(soln.x)
        plane1.N = soln.x[:3]
        plane1.d = soln.x[3]
        plane2.N = soln.x[4:7]
        plane2.d = soln.x[7]

        return plane1, plane2

    plane1, plane2 = fit_2planes(points1, points2)
    print(f"Plane1 N: {plane1.N}, d: {plane1.d}. Error: {plane1.error}")
    print(f"Plane2 N: {plane2.N}, d: {plane2.d}. Error: {plane2.error}")
    return plane1, plane2

def test():
    ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
    last_distances = np.zeros((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    last_sigma = np.ones((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    while True:
        ## 1. Read the data from the serial port
        distances, sigma = read_serial_data(ser, cfg.Sensor["resolution"])
        ## 2. Refine by time
        time_refine_distances, time_refine_sigma = refine_by_time(
            distances, sigma, last_distances, last_sigma
        )
        last_distances = distances
        last_sigma = sigma

        points3D = np.array(
            [
                [i, j, time_refine_distances[i, j]]
                for i in range(cfg.Sensor["resolution"])
                for j in range(cfg.Sensor["resolution"])
            ]
        )

        # if cfg.Code["distance_rectified_fov"]:
        #     points_world = distance_rectified_fov(points3D)

        ## 3. Calculate the gradioents of x and y direction
        grad_y = np.gradient(time_refine_distances, axis=1)
        grad_x = np.gradient(time_refine_distances, axis=0)

        ## 4. Divide the plane into two parts based on the angle
        plane1_index = []
        plane2_index = []
        for i in range(cfg.Sensor["resolution"]):
            for j in range(cfg.Sensor["resolution"]):
                if grad_y[i][j] > 0:
                    plane1_index.append([i, j])
                else:
                    plane2_index.append([i, j])
        plane1_index = np.array(plane1_index)
        plane2_index = np.array(plane2_index)
        ## 5. Outliers detection
        plane1_index = outliers_detection(plane1_index, 4)
        plane2_index = outliers_detection(plane2_index, 4)
        points_plane1 = np.array([points3D[i * 8 + j] for [i, j] in plane1_index])
        points_plane2 = np.array([points3D[i * 8 + j] for [i, j] in plane2_index])
        ## 5. Plane fitting
        if cfg.Code["distance_rectified_fov"]:
            points_plane1 = distance_rectified_fov(np.array(points_plane1))
            points_plane2 = distance_rectified_fov(np.array(points_plane2))
        plane1 = Plane(np.array([0, 0, 1]), 0)
        plane2 = Plane(np.array([0, 0, 1]), 0)
        # plane1, plane2 = two_planes_fitting(points_plane1, points_plane2)
        # ToF_RANSAC will transfer the points to the world coordinate
        plane1 = plane1.ToF_RANSAC(points_plane1, res=cfg.Sensor["resolution"])
        plane2 = plane2.ToF_RANSAC(points_plane2, res=cfg.Sensor["resolution"])
        ## 6. Visualization
        fig = plt.figure(figsize=(14, 7))
        # Transfer the visulaization to the world coordinate
        two_plane_visualization(fig, plane1, plane2, points_plane1, points_plane2)
        plane1.ToF_visualization(
            fig,
            time_refine_distances,
            time_refine_sigma,
            cfg.Sensor["resolution"],
            cfg.Sensor["output_shape"],
        )
        plt.show()
        print(f"Plane1 N: {plane1.N}, d: {plane1.d}, Error: {plane1.error}.")
        print(f"Plane2 N: {plane2.N}, d: {plane2.d}, Error: {plane2.error}.")


if __name__ == "__main__":
    test()
