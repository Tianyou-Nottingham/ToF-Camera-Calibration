import numpy as np
import configs.config as cfg
import serial
from read_data_utils import read_serial_data, visualize2D, normalize, refine_by_time
import cv2
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import time
from TOF_RANSAC import Plane

## For ToF sensor, we think the imaging model is the same as camera: a pinhole camera model.
## Only difference is the resolution.
## For camera and ToF calibration, we can use the same calibration method.
## Detect the corners of the chessboard, and use the corners to calibrate the camera.
## But how to detect the corners of the chessboard in ToF data?
## 1. We need a different-depth chessboard.
## 2. We need to detect the corners in the low-resolution depth image.
## So this code is to detect the corners in the low-resolution depth image.
## Of course, we need a video input or continuous depth image input.


def padding(data, pad_size):
    w, h = data.shape
    pad_data = np.zeros((w + 2 * pad_size, h + 2 * pad_size))
    pad_data[pad_size: w + pad_size, pad_size: h + pad_size] = data
    return pad_data

def tof_to_camera(x, y, image_shape):
    ## transform the ToF image to camera image
    pad_size = image_shape // cfg.Sensor["resolution"]
    return [(x + 1/2) * pad_size, (y + 1/2) * pad_size]

def kmeans_clustering(data, k):
    ## data:[8, 8] depth map
    ## k: number of clusters
    ## return the segmentation line
    ## For the zones divided by the segmentation line, we calculate the mean depth value. And set the new 
    w, h = data.shape
    ## random initialization
    centers = [np.random.choice(w, k), np.random.choice(h, k)]
    ## cluster index and value
    cluster_index = [[] for _ in range(k)]
    cluster_value = [[] for _ in range(k)]
    ## When the cluster is changed, we need to update the cluster index and value. 
    ## And if the cluster is not changed, it means the stable centers have been found.
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(w):
            for j in range(h):
                min_dist = np.inf
                min_index = -1
                dist = []
                ## To prevent the centers are not found, we need to check the number of centers.
                for center in centers:
                    ## Calculate the depth distance for every zone
                        dist.append((data[i, j] - data[round(center[0]), round(center[1])])**2)

                ## Find the closer center
                min_index = np.argmin(dist)
                ## Center zone adds the depth value
                cluster_value[min_index].append(data[i, j])
                ## It means the cluster is changed
                if (i, j) not in cluster_index[min_index]:
                    cluster_changed = True
                    cluster_index[min_index].append((i, j))
                else:
                    continue

                # if min_dist > dist[min_index]:
                #     min_dist = dist[min_index]
                #     if (i, j) not in cluster_index[min_index]:
                #         cluster_index[min_index].append((i, j))
                #     else:
                #         continue                    
        value = {}
        for idx in range(k):
            if cluster_index[idx] == []:
                centers[idx] = [0, 0]
            else:
                centers[idx] = np.mean(cluster_index[idx], axis=0)
            value.update({np.mean(cluster_value[idx], axis=0): cluster_index[idx]})
        
        value = sorted(value.items(), key=lambda x: x[0])
    return list(dict(value).values()) ## return the cluster index

def outliers_detection(data, threshold):
    ## data: cluster index
    ## return the new cluster index
    ## We need to detect the outliers in the cluster index
    ## If the distance between the point and the center is larger than the threshold, we need to remove the point.
    ## And we need to recalculate the center.
    center = np.mean(data, axis=0)
    for i in range(len(data)):
        if np.linalg.norm(np.array(data[i]) - center) > threshold:
            np.delete(data, i)
    return data

def edge_detect(data):
    padding_data = padding(data, 1)
    vertical_edge = np.zeros_like(data)
    horizontal_edge = np.zeros_like(data)
    for i in range(1, data.shape[0] + 1):
        for j in range(1, data.shape[1] + 1):
            ## Sobel operator
            vertical_edge[i - 1, j - 1] = (padding_data[i - 1, j + 1] + 2 * padding_data[i, j + 1] + padding_data[i + 1, j + 1] \
                - padding_data[i - 1, j - 1] - 2 * padding_data[i, j - 1] - padding_data[i + 1, j - 1]) / 6
            horizontal_edge[i - 1, j - 1] = (padding_data[i + 1, j - 1] + 2 * padding_data[i + 1, j] + padding_data[i + 1, j + 1] \
                - padding_data[i - 1, j - 1] - 2 * padding_data[i - 1, j] - padding_data[i - 1, j + 1]) / 6
    return vertical_edge, horizontal_edge

def line_detect(data, threshold):
    vertical_edge, horizontal_edge = edge_detect(data)
    vertical_edge = np.abs(vertical_edge)
    horizontal_edge = np.abs(horizontal_edge)
    verticle_edge_center = []
    horizontal_edge_center = []
    for i in range(vertical_edge.shape[0]):
        for j in range(vertical_edge.shape[1]):
            if data[i, j] == 0:
                continue
            elif vertical_edge[i, j] > threshold:
                verticle_edge_center.append((i, j))
            elif horizontal_edge[i, j] > threshold:
                horizontal_edge_center.append((i, j))
    if len(verticle_edge_center) == 0 or len(horizontal_edge_center) == 0:
        return None, None
    else:
        vertical_lines = np.polyfit(verticle_edge_center[0], verticle_edge_center[1], 1)
        horizontal_lines = np.polyfit(horizontal_edge_center[0], horizontal_edge_center[1], 1)
    return vertical_lines, horizontal_lines


def main():
    ser = serial.Serial(cfg.Serial["port"], cfg.Serial["baudrate"])
    last_distances = np.zeros((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    last_sigma = np.ones((cfg.Sensor["resolution"], cfg.Sensor["resolution"]))
    pad_size = cfg.Sensor["output_shape"][0] // cfg.Sensor["resolution"]
    output_shape = cfg.Sensor["output_shape"]
    points3D = []
    while True:
        ## 1. Read the data from the serial port
        distances, sigma = read_serial_data(ser, cfg.Sensor["resolution"])
        points3D = np.array([[i, j, distances[i, j]] for i in range(cfg.Sensor["resolution"]) for j in range(cfg.Sensor["resolution"])])
        ## 2. Refine by time
        time_refine_distances, time_refine_sigma = refine_by_time(distances, sigma, last_distances, last_sigma)
        print(time_refine_distances)
        last_distances = distances
        last_sigma = sigma
        ## 3. K-means clustering
        points_index = kmeans_clustering(time_refine_distances, 2)
        ## 4. Outliers detection
        points_index = outliers_detection(points_index, 4)
        ## 5. Plane fitting
        points_obstacle = np.array([[i, j, time_refine_distances[i, j]] for [i, j] in points_index[0]])
        points_safe = np.array([[i, j, time_refine_distances[i, j]] for [i, j] in points_index[1]])
        plane_obstacle = Plane(np.array([0, 0, 1]), 0)
        plane_safe = Plane(np.array([0, 0, 1]), 0)
        plane_obstacle, error_obstacle = plane_obstacle.ToF_RANSAC(points_obstacle, res=cfg.Sensor["resolution"])
        plane_safe = plane_safe.ToF_RANSAC(points_safe, res=cfg.Sensor["resolution"])
        centers = [np.mean(points_index[0], axis=0), np.mean(points_index[1], axis=0)]
        ## 6. Visualization
        depth, sigma = visualize2D(time_refine_distances, time_refine_sigma, cfg.Sensor["resolution"], cfg.Sensor["output_shape"])
        # print(f"Obstacle plane N: {plane_obstacle.N}, d: {plane_obstacle.d}.")
        # print(f"Safe plane N: {plane_safe.N}, d: {plane_safe.d}.")


        color_depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)        
        cv2.circle(color_depth, (round(centers[0][1]*pad_size), round(centers[0][0]*pad_size)), 5, (0, 255, 0), -1)
        cv2.putText(color_depth, "Obstacle", (round(centers[0][1]*pad_size), round(centers[0][0]*pad_size)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(color_depth, (round(centers[1][1]*pad_size), round(centers[1][0]*pad_size)), 5, (0, 0, 255), -1)
        cv2.putText(color_depth, "Safe", (round(centers[1][1]*pad_size), round(centers[1][0]*pad_size)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out = np.hstack([color_depth, cv2.applyColorMap(sigma, cv2.COLORMAP_TURBO)])
        cv2.imshow('depth', out)
        cv2.waitKey(1) & 0xFF == ord('q')

        
if __name__ == "__main__":
    main() 