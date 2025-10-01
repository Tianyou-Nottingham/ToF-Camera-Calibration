import cv2
import numpy as np
import pyrealsense2 as rs
import os
import time
import serial
import scipy
from TOF_RANSAC import Plane
from read_data_utils import read_serial_data, refine_by_time, visualize2D
import configs.config as cfg
import matplotlib.pyplot as plt
from obstacle_avoidance import outliers_detection
from utils.distance_rectified_fov import distance_rectified_fov
from two_plane_fit import two_plane_visualization, two_planes_fitting
import re

RGB_path = r"E:\Projects\ToF\ToF\output\RGB1_Color.png"
Depth_path = r"C:\Users\ezxtz6\Pictures\Depth0.png"
Intrinsic = cfg.RealSense["K"]
chessboard_size = [10, 7]


def normalize(value, vmin=0.0, vmax=4.0):
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


def corner_detection(img):
    '''
    棋盘格角点检测
    return ret, img_color, points_w, points_i
    '''
    img_color = img
    Grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objpoints = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objpoints[:, :2] = np.mgrid[
        0 : chessboard_size[0], 0 : chessboard_size[1]
    ].T.reshape(-1, 2)
    points_w = []
    points_i = []
    ret, corners = cv2.findChessboardCorners(Grey, chessboard_size, None)
    print(f"Corner Detection: {ret}")
    if ret:
        cv2.cornerSubPix(
            Grey,
            corners,
            chessboard_size,
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        points_w.append(objpoints)
        points_i.append(corners)
        cv2.drawChessboardCorners(img_color, chessboard_size, corners, ret)
        # cv2.imshow("RGB", img)
        # cv2.imwrite("./RGB_corners.png", img)
        # cv2.waitKey(0)
    return ret, img_color, points_w, points_i

def contours_detection(img_color):
    '''
    靶标轮廓检测
    return ret, img_color, contours
    '''
    imgray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    contour_img = img_color
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("img", thresh)
    # cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 查找靶标最外层轮廓
    if contours is not None:
        for i in range(len(hierarchy[0])):
            if hierarchy[0][i][2] == 2:
                # 只画出最外层的第一级子轮廓
                cv2.drawContours(contour_img, contours, i, (0, 255, 0), 3)
                # 只保存矩形四个顶点坐标
                simplified_contours  = cv2.approxPolyDP(contours[i], 0.02 * cv2.arcLength(contours[i], True), True)
                if len(simplified_contours) == 4:
                    left = simplified_contours[0][0]
                    right = simplified_contours[2][0]
                    top = simplified_contours[1][0]
                    bottom = simplified_contours[3][0]
                box = np.array([left, right, top, bottom])
                print(f"Contours detection: True")
                return True, contour_img, box
    return False, contour_img, None


def find_line(img, corners, chessboard_size):
    '''
    棋盘格线检测
    return img, lines_l, lines_s
    '''
    l, s = chessboard_size
    lines_l = []
    lines_s = []
    for i in range(s):
        line = [corners[i * l], corners[(i + 1) * l - 1]]
        lines_l.append(line)  # 长边
    for j in range(l):
        line = [corners[j], corners[(s - 1) * l + j]]
        lines_s.append(line)  # 短边
    # Draw the lines
    for line in lines_s:
        img = cv2.line(
            img, line[0].astype(np.uint), line[-1].astype(np.uint), (0, 255, 0), 2 # color: green
        )
    for line in lines_l:
        img = cv2.line(
            img, line[0].astype(np.uint), line[-1].astype(np.uint), (0, 0, 255), 2 # color: red
        )
    return img, lines_l, lines_s


def find_infinity_point(img, lines):
    """
    计算两条平行线的无穷远点
    :param img:
    :param lines: 平行线的两个端点
    :return: 无穷远点的齐次坐标 (x, y, 0)

    ## In camera projection space, the intersection is the same as the vanishing point
    ## We can use the cross product to find the intersection
    ## The intersection of two lines is the vanishing point
    """
    w, h = chessboard_size
    lines_projection = []
    for line in lines:
        A = np.array([line[0][0], line[0][1], 1])
        B = np.array([line[-1][0], line[-1][1], 1])
        lines_projection.append(np.cross(A, B))
    ## RANSAC to find the intersection
    ## The intersection is the vanishing point
    best_point = [0, 0, 0]
    best_error = np.inf
    max_iter = 1000
    pretotal = 0
    k = 0
    total_inlier = 0
    while (k < max_iter) and (total_inlier < len(lines_projection) * 2 / 3):
        [A_index, B_index] = np.random.choice(len(lines_projection), 2, replace=False)
        A_line = lines_projection[A_index]
        B_line = lines_projection[B_index]
        point = np.cross(A_line, B_line)
        error = 0
        total_inlier = 0
        for i in range(len(lines_projection)):
            if np.linalg.norm(np.cross(point, lines_projection[i])) < 1e-3:
                total_inlier += 1
            error += np.linalg.norm(np.cross(point, lines_projection[i]))
        if error < best_error:
            best_error = error
            best_point = point
        k += 1
    return best_point / best_point[-1]


def rs_capture(save=True, contours_collection=False):
    # 创建realsense pipeline 以及 serial
    pipeline = rs.pipeline()

    ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
    last_distances = np.zeros((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    last_sigma = np.ones((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    # Create a config并配置要流​​式传输的管道
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)
    # 将depth对齐到color
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 保存的图片和实时的图片界面
    cv2.namedWindow("RealSense live", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("ToF live", cv2.WINDOW_AUTOSIZE)

    if save == True:
        # 按照日期创建文件夹
        save_path = os.path.join(
            os.getcwd(), "calib", time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        )
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            RGB_path = os.path.join(save_path, "color")
            os.mkdir(RGB_path)
            RS_path = os.path.join(save_path, "depth")
            os.mkdir(RS_path)
            ToF_path = os.path.join(save_path, "ToF")
            os.mkdir(ToF_path)
        cv2.namedWindow("save", cv2.WINDOW_AUTOSIZE)

    print(f"Save to {RGB_path}")
    saved_color_image = None  # 保存的临时图片
    saved_depth_mapped_image = None
    saved_ToF_depth = None
    
    
    # 主循环
    try:
        while True:
            saved_count = 0
            #### 1.1 Read RealSense data ####
            frames = pipeline.wait_for_frames()
            profile = pipeline.get_active_profile()

            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            # RealSense倒着放置的，需要上下左右翻转
            color_image = cv2.flip(color_image, -1)

            depth_data = np.asanyarray(aligned_depth_frame.get_data())
            depth_data = cv2.flip(depth_data, -1)
            scaled_depth_data = depth_data * depth_scale

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_image = cv2.flip(depth_image, -1)
            scaled_depth_image = depth_image * depth_scale

            depth_mapped_image = cv2.applyColorMap(
                normalize(scaled_depth_image, vmax=1),
                cv2.COLORMAP_MAGMA,
            )

            #### 1.2 Read ToF data ####
            distances, sigma = read_serial_data(ser, cfg.Sensor["resolution"])
            time_refine_distances, time_refine_sigma = refine_by_time(
                distances, sigma, last_distances, last_sigma
            )
            last_distances = distances
            last_sigma = sigma

            ToF_depth_map, ToF_sigma = visualize2D(
                time_refine_distances,
                sigma,
                cfg.Sensor["resolution"],
                cfg.Sensor["output_shape"],
            )
            ToF_depth_img = cv2.applyColorMap(
                normalize(ToF_depth_map, vmax=1000), cv2.COLORMAP_MAGMA
            )

            ##### 2. Image and ToF data processing #####
            #### 2.1 ToF data processing ####
            grad_x = np.gradient(time_refine_distances, axis=1)
            grad_y = np.gradient(time_refine_distances, axis=0)
            plane1_index = []
            plane2_index = []
            for i in range(cfg.Sensor["resolution"]):
                for j in range(cfg.Sensor["resolution"]):
                    if grad_x[i][j] > 0:
                        plane1_index.append([i, j])
                    else:
                        plane2_index.append([i, j])
            plane1_index = np.array(plane1_index)
            plane2_index = np.array(plane2_index)
            plane1_index = outliers_detection(plane1_index, 4)
            plane2_index = outliers_detection(plane2_index, 4)
            plane1_points = np.array(
                [
                    [i, j, time_refine_distances[i, j]]
                    for i, j in plane1_index
                    if time_refine_distances[i, j] > 1.5
                    and time_refine_distances[i, j] < 1000
                ]
            )
            plane2_points = np.array(
                [
                    [i, j, time_refine_distances[i, j]]
                    for i, j in plane2_index
                    if time_refine_distances[i, j] > 1.5
                    and time_refine_distances[i, j] < 1000
                ]
            )

            #### 2.2 Realsense processing ####
            color_blur = cv2.GaussianBlur(color_image, (0, 0), 5)
            color_usm = cv2.addWeighted(color_image, 1.5, color_blur, -0.5, 0)
            corner_detection_ret, corner_detection_img, points_w, points_i = (
                corner_detection(color_usm)
            )
             ## 棋盘格尺寸15mm,points_w所有值×15mm
            points_w = np.array(points_w) * 15
            points_i = np.array(points_i) 
            points_w = points_w.reshape((-1, 3))
            points_i = points_i.reshape((-1, 2))
            # print(f"Shape of w is {points_w.shape}, shape of i is {points_i.shape}")

            if corner_detection_ret == True and contours_collection == True:
                ##### 检测轮廓 #####
                ret_contour, contours_detection_img, contour = (contours_detection(color_usm))
                if ret_contour == True:
                    # cv2.drawContours(corner_detection_img, contour, -1, (0, 255, 0), 2)
                    ##### 3 Prepare ToF plane fitting #####
                    plane1 = Plane(np.array([1, 0.5, 0]), 50)
                    plane2 = Plane(np.array([-1, 0.5, 0]), 50)
                    plane3 = Plane(np.array([0, 1, 0.5]), 50)
                    ##### 4. Visualization #####
                    cv2.imshow(
                        "RealSense live",
                        np.hstack((contours_detection_img, depth_mapped_image)),
                    )

                    cv2.imshow("ToF live", ToF_depth_img)
                    # key = cv2.waitKey(30)

                    #### 3.1 Plane fitting ####
                    if cfg.Code["distance_rectified_fov"]:
                        points1 = distance_rectified_fov(plane1_points)
                        points2 = distance_rectified_fov(plane2_points)
                    plane1 = plane1.ToF_RANSAC(points1, res=cfg.Sensor["resolution"])
                    plane2 = plane2.ToF_RANSAC(points2, res=cfg.Sensor["resolution"])

                    fig = plt.figure(figsize=(14, 7))

                    plane1.ToF_visualization(
                    fig,
                    time_refine_distances,
                    time_refine_sigma,
                    cfg.Sensor["resolution"],
                    cfg.Sensor["output_shape"],
                    )

                    two_plane_visualization(fig, plane1, plane2, points1, points2)

                    ### 3.2 Realsense vanishing point calculation ####
                    ret, rvec, tvec = cv2.solvePnP(
                        points_w.astype("float32"),
                        points_i.astype("float32"),
                        Intrinsic,
                        None,
                    )
                    print(f"rvec: {rvec}, tvec: {tvec}")
                    R, _ = cv2.Rodrigues(rvec)
                    points_c = R @ points_w.T + tvec
                    points_c = points_c.T
                    plane3 = plane3.fit_plane(points_c)
                    # print(f"Plane3: N: {plane3.N}, d:{plane3.d}, error:{plane3.error}")
                    line_image, lines_w, lines_l = find_line(
                        color_usm, points_i, chessboard_size
                    )
                    vanishing_point_w = find_infinity_point(color_usm, lines_w)
                    vanishing_point_l = find_infinity_point(color_usm, lines_l)
                    
                    # Save the plane fitting result and vanishing points to .txt file
                    def on_key_press(event):
                        if event.key == "p":
                            txt_name = os.path.join(save_path, "plane_fitting.txt")
                            with open(txt_name, "a") as f:
                                f.write(
                                    f"Vanishing Point W: {vanishing_point_w}, L: {vanishing_point_l}.\n"
                                    f"Plane1: N: {plane1.N}, d:{plane1.d}, error:{plane1.error}; \n"
                                    f"Plane2: N: {plane2.N}, d:{plane2.d},error:{plane2.error}; \n"
                                    f"Plane3c: N: {plane3.N}, d:{plane3.d},error:{plane3.error}. \n"
                                    f"Contours: {contour}\n"
                                ) 
                            saved_color_image = color_image
                            saved_depth_mapped_image = depth_mapped_image
                            print(f"Save to {RGB_path}")
                            # 彩色图片保存为png格式
                            cv2.imwrite(
                                os.path.join(
                                    RGB_path, time.strftime("%H_%M_%S", time.localtime()) + ".png"
                                ),
                                saved_color_image,
                            )
                            # 深度信息由采集到的float16直接保存为npy格式
                            np.save(
                                os.path.join((RS_path), "{}".format(time.strftime("%H_%M_%S", time.localtime()))),
                                depth_data,
                            )
                            np.save(
                                os.path.join((ToF_path), "{}".format(time.strftime("%H_%M_%S", time.localtime()))),
                                time_refine_distances,
                            )
                            cv2.imshow(
                                "save", np.hstack((saved_color_image, saved_depth_mapped_image))
                            )
                            pass
                        else:
                            pass

                    # 将回调函数与图形对象绑定
                    fig.canvas.mpl_connect("key_press_event", on_key_press)
                    plt.show()

                else:
                    print(f"Contours detection: False")
                    cv2.imshow(
                        "RealSense live",
                        np.hstack((corner_detection_img, depth_mapped_image)),
                    )
                    cv2.imshow("ToF live", ToF_depth_img)
                    # key = cv2.waitKey(30)

            elif corner_detection_ret == True and contours_collection == False:
                ## 只采集R的数据 ##
                ##### 3 Prepare ToF plane fitting #####
                plane1 = Plane(np.array([1, 0.5, 0]), 50)
                plane2 = Plane(np.array([-1, 0.5, 0]), 50)
                ##### 4. Visualization #####
                cv2.imshow(
                    "RealSense live",
                    np.hstack((corner_detection_img, depth_mapped_image)),
                )

                cv2.imshow("ToF live", ToF_depth_img)
                # key = cv2.waitKey(30)

                #### 3.1 Plane fitting ####
                if cfg.Code["distance_rectified_fov"]:
                    points1 = distance_rectified_fov(plane1_points)
                    points2 = distance_rectified_fov(plane2_points)
                plane1 = plane1.ToF_RANSAC(points1, res=cfg.Sensor["resolution"])
                plane2 = plane2.ToF_RANSAC(points2, res=cfg.Sensor["resolution"])

                fig = plt.figure(figsize=(14, 7))

                plane1.ToF_visualization(
                    fig,
                    time_refine_distances,
                    time_refine_sigma,
                    cfg.Sensor["resolution"],
                    cfg.Sensor["output_shape"],
                )

                two_plane_visualization(fig, plane1, plane2, points1, points2)

                ### 3.2 Realsense vanishing point calculation ####
                line_image, lines_l, lines_s = find_line(
                    color_usm, points_i, chessboard_size
                )
                vanishing_point_l = find_infinity_point(color_usm, lines_l)
                vanishing_point_s = find_infinity_point(color_usm, lines_s)
                cv2.imshow(
                    "RealSense live",
                    np.hstack((corner_detection_img, depth_mapped_image)),
                )
                cv2.imshow("ToF live", ToF_depth_img)
                # key = cv2.waitKey(30)
                def on_key_press(event):
                        if event.key == "p":    
                            txt_name = os.path.join(save_path, "plane_fitting.txt")
                            with open(txt_name, "a") as f:
                                f.write(
                                    f"Vanishing Point L: {vanishing_point_l}, S: {vanishing_point_s}.\n"
                                    f"Plane1: N: {plane1.N}, d:{plane1.d}, error:{plane1.error}; \n"
                                    f"Plane2: N: {plane2.N}, d:{plane2.d}, error:{plane2.error}; \n"
                                )

                            saved_color_image = color_image
                            saved_depth_mapped_image = depth_mapped_image
                            print(f"Save to {RGB_path}")
                            # 彩色图片保存为png格式
                            cv2.imwrite(
                                os.path.join(
                                    RGB_path, time.strftime("%H_%M_%S", time.localtime()) + ".png"
                                ),
                                saved_color_image,
                            )
                            # 深度信息由采集到的float16直接保存为npy格式
                            np.save(
                                os.path.join((RS_path), "{}".format(time.strftime("%H_%M_%S", time.localtime()))),
                                depth_data,
                            )
                            np.save(
                                os.path.join((ToF_path), "{}".format(time.strftime("%H_%M_%S", time.localtime()))),
                                time_refine_distances,
                            )
                            cv2.imshow(
                                "save", np.hstack((saved_color_image, saved_depth_mapped_image))
                            )
                            pass
                        else:
                            pass

                # 将回调函数与图形对象绑定
                fig.canvas.mpl_connect("key_press_event", on_key_press)
                plt.show()
                
                    
            ##### 4. Visualization #####
            else:
                print(f"Contours detection: False")
                cv2.imshow(
                    "RealSense live", np.hstack((color_image, depth_mapped_image))
                )
                cv2.imshow("ToF live", ToF_depth_img)
                # key = cv2.waitKey(30)

            key = cv2.waitKey(30)

            if save == True:
                # s 保存图片
                if key & 0xFF == ord("s"):
                    saved_color_image = color_image
                    saved_depth_mapped_image = depth_mapped_image

                    # 彩色图片保存为png格式
                    cv2.imwrite(
                        os.path.join(
                            (save_path), "color", "{}.png".format(saved_count)
                        ),
                        saved_color_image,
                    )
                    # 深度信息由采集到的float16直接保存为npy格式
                    np.save(
                        os.path.join((save_path), "depth", "{}".format(saved_count)),
                        depth_data,
                    )
                    np.save(
                        os.path.join((save_path), "ToF", "{}".format(saved_count)),
                        time_refine_distances,
                    )
                    saved_count += 1
                    cv2.imshow(
                        "save", np.hstack((saved_color_image, saved_depth_mapped_image))
                    )

            # q 退出
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


def main():
    Rgb = cv2.imread(RGB_path)
    # 上下左右翻转图片
    Rgb = cv2.flip(Rgb, -1)
    Rgb_blur = cv2.GaussianBlur(Rgb, (0, 0), 5)
    usm = cv2.addWeighted(Rgb, 1.5, Rgb_blur, -0.5, 0)
    cv2.imshow("RGB", usm)
    cv2.waitKey(0)
    print(Rgb.shape)
    ## 1. Corner Detection
    points_w, points_i = corner_detection(usm)

    ## 2. Find the lines
    line_img, lines_horizontal, lines_vertical = find_line(
        usm, points_i[0], chessboard_size
    )
    cv2.imshow("RGB", line_img)
    cv2.imwrite("./RGB_lines.png", line_img)
    cv2.waitKey(0)

    ## 3. Find the intersection
    # change the points into projection space
    # points_i = np.array(points_i).reshape((chessboard_size[1], chessboard_size[0], 2))
    vanishing_point_horizontal = find_infinity_point(Rgb, lines_horizontal)
    vanishing_point_vertical = find_infinity_point(Rgb, lines_vertical)
    if (
        vanishing_point_horizontal[0] < Rgb.shape[0]
        and vanishing_point_horizontal[1] < Rgb.shape[1]
    ):
        cv2.circle(
            Rgb,
            (int(vanishing_point_horizontal[0]), int(vanishing_point_horizontal[1])),
            5,
            (0, 255, 0),  ## BGR
            -1,
        )
    if (
        vanishing_point_vertical[0] < Rgb.shape[0]
        and vanishing_point_vertical[1] < Rgb.shape[1]
    ):
        cv2.circle(
            Rgb,
            (int(vanishing_point_vertical[0]), int(vanishing_point_vertical[1])),
            5,
            (0, 0, 255),
            -1,
        )
    cv2.imshow("RGB", Rgb)
    cv2.imwrite("./RGB_vanishing_point.png", Rgb)
    cv2.waitKey(0)
    print(f"Vanishing Point Horizontal: {vanishing_point_horizontal}")
    print(f"Vanishing Point Vertical: {vanishing_point_vertical}")


def read_N_and_Vp(file_path):
    vp_l = []
    vp_s = []
    plane1_N = []
    plane1_d = []
    plane2_N = []
    plane2_d = []
    plane3_N = []
    plane3_d = []
    line13 = []
    line23 = []
    with open(file_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            if "Vanishing Point" in line:
                temp = re.split("\[|\]", line)
                vp_l.append([float(i) for i in temp[1].split()])
                vp_s.append([float(i) for i in temp[3].split()])
            elif "Plane1" in line:
                temp = re.split("\[|\]|:", line)
                plane1_N.append([float(i) for i in temp[3].split()])
                plane1_d.append(float(re.split(",", temp[5])[0]))
            elif "Plane2" in line:
                temp = re.split("\[|\]|:", line)
                plane2_N.append([float(i) for i in temp[3].split()])
                plane2_d.append(float(re.split(",", temp[5])[0]))
            elif "Plane3c" in line:
                temp = re.split("\[|\]|:", line)
                plane3_N.append([float(i) for i in temp[3].split()])
                plane3_d.append(float(re.split(",", temp[5])[0]))
            elif "Contours" in line:
                # temp = re.split("\[|\]|:", line).strip()
                numbers = line.replace("Contours:", "").replace("[", "").replace("]", " ").split()
                numbers = np.array(list(map(int, numbers))).reshape(-1,2)  # 转换为整数
                # 寻找最靠近原点的两条直线
                up = numbers[np.argmin(numbers[:, 1])]
                left = numbers[np.argmin(numbers[:, 0])]
                right = numbers[np.argmax(numbers[:, 0])]
                # 转换到归一化平面
                ux, uy = Intrinsic[0, 2], Intrinsic[1, 2]
                fx, fy = Intrinsic[0, 0], Intrinsic[1, 1]
                up = [(up[0] - ux) / fx, (up[1] - uy) / fy]
                left = [(left[0] - ux) / fx, (left[1] - uy) / fy]
                right = [(right[0] - ux) / fx, (right[1] - uy) / fy]
                points13 = [up, left]
                points23 = [up, right]

                # 直线方程ax+by+c=0 (a,b,c)
                a13 = points13[1][1] - points13[0][1]
                b13 = points13[0][0] - points13[1][0]
                c13 = points13[0][1] * points13[1][0] - points13[0][0] * points13[1][1]
                a23 = points23[1][1] - points23[0][1]
                b23 = points23[0][0] - points23[1][0]
                c23 = points23[0][1] * points23[1][0] - points23[0][0] * points23[1][1]
                line13.append([a13, b13, c13])
                line23.append([a23, b23, c23])
                                    
    vp_l = np.array(vp_l)
    vp_s = np.array(vp_s)
    plane1_N = np.array(plane1_N)
    plane1_d = np.array(plane1_d)
    plane2_N = np.array(plane2_N)
    plane2_d = np.array(plane2_d)
    plane3_N = np.array(plane3_N)
    plane3_d = np.array(plane3_d)
    line13 = np.array(line13)
    line23 = np.array(line23)
    return vp_l, vp_s, plane1_N, plane1_d, plane2_N, plane2_d, plane3_N, plane3_d, line13, line23


def calib_R(cfg, Vpl, Vps, N1, N2):
    """
    通过N和Vp计算相机内参
    """
    # 交换N的x y
    N1 = N1[:, [1, 0, 2]]
    N2 = N2[:, [1, 0, 2]]
    Vpl = np.array(Vpl.T)
    Vps = np.array(Vps.T)
    K = Intrinsic
    K_inv = np.linalg.inv(K)
    K_Vpl = K_inv @ Vpl
    K_Vps = K_inv @ Vps
    d1_orient_vec = K_Vpl / np.linalg.norm(K_Vpl, axis=0)
    d2_orient_vec = K_Vps / np.linalg.norm(K_Vps, axis=0)
    d = np.hstack((d1_orient_vec, d2_orient_vec))
    # print(f"d shape: {d.shape}")
    N = np.vstack((N1, N2))
    # print(f"N shape: {N.shape}")
    S = N.T @ d.T
    # print(f"S shape: {S.shape}")
    U, S_, V = np.linalg.svd(S)
    R = V.T @ U.T #@ R_reverse
    r0 = R[:, 0]/np.linalg.norm(R[:, 0])
    r1 = R[:, 1]/np.linalg.norm(R[:, 1])
    r2 = np.cross(r0, r1)
    R = np.vstack((r0, r1, r2)).T
    # 矫正R的行列式为1
    if np.linalg.det(R) < 0:
        R[:, -1] = -R[:, -1]
    print(f"R: {R}")
    return R


def calib_T(cfg, R, N1, d1, N2, d2, N3, d3, line13, line23):
    """
    Here, N is a nx3 vector, d is a n*1 scalar.
    """
    # 交换N的x y
    N1 = N1[:, [1, 0, 2]].T
    N2 = N2[:, [1, 0, 2]].T
    N3 = N3.T
    K = Intrinsic
    np_13 = line13 / np.linalg.norm(line13, axis=1).reshape(-1, 1)
    np_23 = line23 / np.linalg.norm(line23, axis=1).reshape(-1, 1)

    N1_c = np.array(R @ N1)  ## 3xn
    # d1_c = d1 - N1_c @ t ## here contains the translation, which is unknown
    N2_c = np.array(R @ N2)  ## 3xn

    t_A = []
    t_B = []
    for i in range(len(N1)):
        N_i = np.array([N1_c[:, i].T, N2_c[:, i].T, N3[:,i]])
        M_i = np.array([N1_c[:, i].T, N2_c[:, i].T, [0, 0, 0]])
        A = np_13[i,:] @ np.linalg.inv(N_i) @ M_i
        t_A.append(A)
        d_i = np.array([d1[i], d2[i], d3[i]]).reshape((-1, 1))
        b = np_13[i,:] @ np.linalg.inv(N_i) @ d_i
        t_B.append(b)
        # 再算line23
        A = np_23[i,:] @ np.linalg.inv(N_i) @ M_i
        t_A.append(A)
        b = np_23[i,:] @ np.linalg.inv(N_i) @ d_i
        t_B.append(b)

    t_A = np.array(t_A).reshape((-1, 3))  # 3n*3
    t_B = np.array(t_B).reshape((-1, 1))  # 3n*1

    t = np.linalg.lstsq(t_A, t_B, rcond=None)[0]
    print(f"t: {t}")
    return t


def find_contours_TEST():
    # 创建realsense pipeline 以及 serial
    pipeline = rs.pipeline()
    # Create a config并配置要流​​式传输的管道
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)
    # 将depth对齐到color
    align_to = rs.stream.color
    align = rs.align(align_to)
    cv2.namedWindow("RealSense live", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            #### 1.1 Read RealSense data ####
            frames = pipeline.wait_for_frames()
            profile = pipeline.get_active_profile()

            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue
            depth_data = np.asanyarray(aligned_depth_frame.get_data(), dtype="float16")
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            scaled_depth_image = depth_image * depth_scale
            color_image = np.asanyarray(color_frame.get_data())
            depth_mapped_image = cv2.applyColorMap(
                normalize(scaled_depth_image),
                cv2.COLORMAP_MAGMA,
            )
            ### 2.2 Realsense processing ####
            color_blur = cv2.GaussianBlur(color_image, (0, 0), 5)
            color_usm = cv2.addWeighted(color_image, 1.5, color_blur, -0.5, 0)
            corner_detection_ret, corner_detection_img, points_w, points_i = (
                corner_detection(color_usm)
            )
            points_w, points_i = np.array(points_w), np.array(points_i)
            points_w = points_w.reshape((-1, 3))
            points_i = points_i.reshape((-1, 2))
            # print(f"Shape of w is {points_w.shape}, shape of i is {points_i.shape}")
            if corner_detection_ret == True:
                imgray = cv2.cvtColor(color_usm, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # cv2.imshow("img", thresh)
                # cv2.waitKey(0)
                contours, hierarchy = cv2.findContours(
                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                # 查找最外层轮廓
                if contours is not None:
                    for i in range(len(hierarchy[0])):
                        if hierarchy[0][i][2] == 2:
                            # 只画出最外层的第一级子轮廓
                            cv2.drawContours(color_usm, contours, i, (0, 255, 0), 2)
                            # 只保存矩形四个顶点坐标
                            simplified_contours  = cv2.approxPolyDP(contours[i], 0.02 * cv2.arcLength(contours[i], True), True)
                            left = simplified_contours[0][0]
                            right = simplified_contours[2][0]
                            top = simplified_contours[1][0]
                            bottom = simplified_contours[3][0]
                            box = np.array([left, right, top, bottom])
                            print(f"Box: {box}")
                    cv2.imshow(
                        "RealSense live",
                        np.hstack((corner_detection_img, depth_mapped_image)),
                        )
                    key = cv2.waitKey(30)

            else:
                cv2.imshow(
                        "RealSense live", np.hstack((color_image, depth_mapped_image))
                    )
                key = cv2.waitKey(30)
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


if __name__ == "__main__":
    rs_capture(save=True, contours_collection=True)

    # file_path = r"E:\Projects\ToF\ToF\calib\2025_02_18_16_34_52\plane_fitting.txt"
    # Vpl, Vps, N1, d1, N2, d2, N3, d3, line13, line23 = read_N_and_Vp(file_path)
    # # print(f"Vp1: {Vp1}\n, Vp2: {Vp2}\n, N1: {N1}\n, d1: {d1}\n, N2: {N2}\n, d2: {d2}\n, N3: {N3}\n, d3: {d3}\n, line13: {line13}\n, line23: {line23}\n")
    # R = calib_R(cfg, Vpl, Vps, N1, N2)
    # R = np.array([[0.96, 0.005, -0.279], 
    #               [-0.15, 0.834, -0.529], 
    #               [-0.23, 0.552, 0.802]])    # R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    # t = calib_T(cfg, R, N1, d1, N2, d2, N3, d3, line13, line23)

    # find_contours_TEST()
