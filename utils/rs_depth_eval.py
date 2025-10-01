import numpy as np
import scipy
import scipy.optimize
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path as osp
import os
from scipy.interpolate import interp1d

K = np.array(
        [[378.282, 0, 323.395],
        [0, 378.282, 236.58],
        [0, 0, 1],]
    )

def Guassian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2)))

def transform_image(image, K, R, t):
    ## deploy R, t to the image

    K_inv = np.linalg.inv(K)
    h,w = image.shape[:2]
    # 生成所有像素点的坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    ones = np.ones_like(u)
    
    # 构造像素齐次坐标 (u, v, 1)
    pixel_coords = np.stack((u, v, ones), axis=-1).reshape(-1, 3).T  # (3, N)
    # 获取深度信息，并展平为向量
    depth_flat = image.flatten()

    # 计算相机坐标 (X, Y, Z)
    camera_coords = K_inv @ pixel_coords * depth_flat  # (3, N)
    
    # 进行旋转和平移变换
    transformed_coords = R @ (camera_coords - t)  # (3, N)

    # 重新投影到新的像素坐标
    new_pixel_coords = K @ transformed_coords  # (3, N)
    new_pixel_coords /= new_pixel_coords[2, :]  # 归一化

    # 取整数像素位置
    new_u = new_pixel_coords[0, :].reshape(h, w).astype(np.float32)
    new_v = new_pixel_coords[1, :].reshape(h, w).astype(np.float32)
    new_depth = transformed_coords[2, :].reshape(h, w)  # 取 Z' 分量
    # 进行图像重映射
    transformed_image = cv2.remap(new_depth, new_u, new_v, cv2.INTER_LINEAR)
    # for i in range(h):
    #     for j in range(w):
    #         if new_u[i, j] < 0 or new_u[i, j] >= w or new_v[i, j] < 0 or new_v[i, j] >= h:
    #             transformed_image[i, j] = image[i, j]
    return transformed_image
    


def ToF_to_image(ToF_depth):
    N = ToF_depth.shape[0]
    out_res = 480
    out = np.zeros((out_res, out_res))
    for i in range(N):
        for j in range(N):
            out[i*60:(i+1)*60, j*60:(j+1)*60] = ToF_depth[i][j]
    return out


def depth_process(rs_depth_path, ToF_depth_path, R, t):
    rs_depth = np.load(rs_depth_path)
    ToF_depth = np.load(ToF_depth_path)
    rs_depth = rs_depth.astype(np.float32)
    ToF_depth = ToF_depth.astype(np.float32)
    # print("ToF_depth: ", ToF_depth)
    # print("rs_depth: ", rs_depth)

    # fig, axes = plt.subplots(2, 2, figsize=(10, 10))  
    # fig.subplots_adjust(wspace=0.128, hspace=0.12)  # 调整子图之间的间距
    # plt.subplot(2, 2, 1)
    # plt.axis('off')
    # plt.title("ToF Depth Map")
    # im1 = plt.imshow(ToF_to_image(ToF_depth), cmap='magma')
    
    # im1.set_clim(0, 600)
    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes("right", size="5%", pad=0.05)  # 右侧添加 colorbar
    # plt.colorbar(im1, cax=cax)  # 绑定 colorbar
    # plt.subplot(2, 2, 2)
    # plt.axis('off')
    # plt.title("RealSense Depth Map--Before Calibration")
    # im2 = plt.imshow(rs_depth, cmap='magma', )
    # im2.set_clim(0, 600)
    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes("right", size="5%", pad=0.05)  # 右侧添加 colorbar
    # plt.colorbar(im2, cax=cax)  # 绑定 colorbar

    # plt.tight_layout()  # 自动调整布局
    # plt.show()
  
    
    if R is not None:
        calib_depth = transform_image(rs_depth, K, R, t)
    else:
        calib_depth = rs_depth

    calib_depth = calib_depth[:, 80:560]
    rs_zones_calibrated = np.zeros((8, 8))
    rs_zones = np.zeros((8, 8))
    # 将rs深度图划分为8*8个小块，每个小块内统计距离，拟合曲线，得到期望深度
    for i in range(8):
        for j in range(8):
            rs_depth_calibrated_block = calib_depth[i*60:(i+1)*60, j*60:(j+1)*60]
            rs_depth_block = rs_depth[i*60:(i+1)*60, j*60:(j+1)*60]
            bins = np.arange(0, 700, 1)
            calibrated_hist, calibrated_bin_edges = np.histogram(rs_depth_calibrated_block, bins=bins)
            hist, bin_edges = np.histogram(rs_depth_block, bins=bins)
            calibrated_most_freq_depth = calibrated_bin_edges[np.argmax(calibrated_hist)]
            most_freq_depth = bin_edges[np.argmax(hist)]
            calibrated_param, calibrated_covar = scipy.optimize.curve_fit(Guassian, calibrated_bin_edges[:-1], calibrated_hist, p0=[calibrated_most_freq_depth, 5], maxfev=5000)
            mu_depth_calibrated, sigma_calibrated = calibrated_param
            param, covar = scipy.optimize.curve_fit(Guassian, bin_edges[:-1], hist, p0=[most_freq_depth, 5], maxfev=5000)
            mu_depth, sigma = param
            rs_zones_calibrated[i, j] = calibrated_most_freq_depth
            rs_zones[i, j] = most_freq_depth
    # print("rs_zones: ", rs_zones)
    # rs_zones_image = cv2.applyColorMap(ToF_to_image(rs_zones), cv2.COLORMAP_JET)
    # calib_ToF_image = cv2.applyColorMap(ToF_to_image(ToF_depth), cv2.COLORMAP_JET)

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 
    # plt.subplot(1, 2, 1)
    # plt.axis('off')
    # plt.title("ToF Depth Map")
    # im1 = plt.imshow(ToF_to_image(ToF_depth), cmap='magma')
    # im1.set_clim(0, 600)
    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes("right", size="5%", pad=0.05)  # 右侧添加 colorbar
    # plt.colorbar(im1, cax=cax)  # 绑定 colorbar

    # plt.subplot(2, 2, 3)
    # plt.axis('off')
    # plt.title("RealSense Depth Map--After Calibration")
    # im2 = plt.imshow(calib_depth, cmap='magma')

    # im2.set_clim(0, 600)
    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes("right", size="5%", pad=0.05)  # 右侧添加 colorbar
    # plt.colorbar(im2, cax=cax)  # 绑定 colorbar

    # # plt.tight_layout()  # 自动调整布局
    # plt.show()
 
    return rs_zones_calibrated, rs_zones, ToF_depth, calib_depth

def eval(rs_calibrated, rs, ToF):
    rs_calibrated = rs_calibrated.flatten()
    rs = rs.flatten()
    ToF = ToF.flatten()
    diff_calibrated = []
    diff = []
    for i in range(len(rs)):
        if rs[i] > 100 and rs[i] < 700 and ToF[i] > 100 and ToF[i] < 700:
            diff.append(np.abs(rs[i] - ToF[i]))
        if rs_calibrated[i] > 100 and rs_calibrated[i] < 700 and ToF[i] > 100 and ToF[i] < 700:
            diff_calibrated.append(np.abs(rs_calibrated[i] - ToF[i]))
    diff_calibrated = np.array(diff_calibrated)
    diff = np.array(diff)
    
    rms_diff_calibrated = np.sqrt(np.mean(diff_calibrated**2))
    mean_diff_calibrated = np.mean(diff_calibrated)
    std_diff_calibrated = np.std(diff_calibrated)
    rms_diff = np.sqrt(np.mean(diff**2))
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    return mean_diff_calibrated, rms_diff_calibrated, std_diff_calibrated, mean_diff, rms_diff, std_diff

def plot_func(rs_before, rs_after, ToF_data, error_calibrated, error):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(wspace=0.2, hspace=0.12)  # 调整子图之间的间距
    plt.subplot(2, 2, 1)
    plt.axis('off')
    plt.title("ToF Depth Map")
    im1 = plt.imshow(ToF_to_image(ToF_data), cmap='magma')
    im1.set_clim(0, 600)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)  # 右侧添加 colorbar
    plt.colorbar(im1, cax=cax)  # 绑定 colorbar
    plt.subplot(2, 2, 2)
    plt.axis('off')
    plt.title("RealSense Depth Map--Before Calibration")
    im2 = plt.imshow(rs_before, cmap='magma', )
    im2.set_clim(0, 600)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)  # 右侧添加 colorbar
    plt.colorbar(im2, cax=cax)  # 绑定 colorbar
    plt.subplot(2, 2, 3)
    plt.axis('off')
    plt.title("RealSense Depth Map--After Calibration")
    im3 = plt.imshow(rs_after, cmap='magma')
    im3.set_clim(0, 600)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)  # 右侧添加 colorbar
    plt.colorbar(im3, cax=cax)  # 绑定 colorbar

    plt.subplot(2, 2, 4)
    # plt.plot(error_calibrated["mean_diff"], 'b')
    after_cubic_interp = interp1d(range(len(error_calibrated["rms_diff"])), error_calibrated["rms_diff"], kind='cubic')
    before_cubic_interp = interp1d(range(len(error["rms_diff"])), error["rms_diff"], kind='cubic')
    xnew = np.linspace(0, len(error_calibrated["rms_diff"])-1, num=100, endpoint=True)
    plt.plot(xnew, after_cubic_interp(xnew), 'mediumseagreen', label="After calibration")
    plt.plot(xnew, before_cubic_interp(xnew), 'mediumseagreen', linestyle='--', label="Before calibration")
    # plt.plot(error_calibrated["rms_diff"], 'g', label="After calibration")
    # plt.plot(error["mean_diff"], 'b', linestyle='--')
    # plt.plot(error["rms_diff"], 'g', linestyle='--', label="Before calibration")
    plt.xlabel("Image No.")
    plt.ylabel("RMS Error(mm)")
    # plt.legend(["--: Before calibration", "-: After calibration"])
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    R = np.array([[ 0.9812677,  -0.04414566, -0.18752294],
                [-0.00818207,  0.96296267, -0.26951058],
                [ 0.19247532,  0.26599635,  0.94456296]])
    Unity = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    t = np.array([[45.50863735],
                [39.29209392],
                [-137.2972859]])
    t0 = np.array([[0],
                [0],
                [0]])
    Error_calibrated = {"mean_diff":[], "rms_diff": [], "std_diff": []}
    Error = {"mean_diff":[], "rms_diff": [], "std_diff": []}
    rs_path = r"E:\Projects\ToF\ToF\calib\2025_02_18_16_34_52\depth"
    ToF_path = r"E:\Projects\ToF\ToF\calib\2025_02_18_16_34_52\ToF"
    for name in os.listdir(rs_path):
        rs_depth_path = osp.join(rs_path, name)
        ToF_depth_path = osp.join(ToF_path, name)
        rs_zones_calibrated, rs_zones, ToF_depth, calib_depth = depth_process(rs_depth_path, ToF_depth_path, R, t)
        # print("rs_zones_calibrated: ", rs_zones_calibrated)
        # print("rs_zones: ", rs_zones)
        # print("ToF_depth: ", ToF_depth)
        mean_diff_calibrated, rms_diff_calibrated, std_diff_calibrated, mean_diff, rms_diff, std_diff = eval(rs_zones_calibrated, rs_zones, ToF_depth)
        Error_calibrated["mean_diff"].append(mean_diff_calibrated)
        Error_calibrated["rms_diff"].append(rms_diff_calibrated)
        Error_calibrated["std_diff"].append(std_diff_calibrated)
        Error["mean_diff"].append(mean_diff)
        Error["rms_diff"].append(rms_diff)
        Error["std_diff"].append(std_diff)

    print("mean_diff_calibrated: ", np.mean(Error_calibrated["mean_diff"]))
    print("rms_diff_calibrated: ", np.mean(Error_calibrated["rms_diff"]))
    print("std_diff_calibrated: ", np.mean(Error_calibrated["std_diff"]))
    print("mean_diff: ", np.mean(Error["mean_diff"]))
    print("rms_diff: ", np.mean(Error["rms_diff"]))
    print("std_diff: ", np.mean(Error["std_diff"]))

    plot_rs_path = r"E:\Projects\ToF\ToF\calib\2025_02_18_16_19_39\depth\16_31_02.npy"
    plot_ToF_path = r"E:\Projects\ToF\ToF\calib\2025_02_18_16_19_39\ToF\16_31_02.npy"
    rs_before = np.load(plot_rs_path)
    rs_zones_calibrated, rs_zones, ToF_depth, rs_after = depth_process(plot_rs_path, plot_ToF_path, R, t)
    ToF_data = np.load(plot_ToF_path)
    plot_func(rs_before, rs_after, ToF_data, Error_calibrated, Error)