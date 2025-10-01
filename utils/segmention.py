import cv2
import numpy as np

NEIGHBORHOOD_4 = True
NEIGHBORHOOD_8 = False
OFFSETS_4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]
OFFSETS_8 = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
THRESHOLD = 50 ## 50mm

def reorganize(img: np.array):
    index_map = []
    points = []
    index = -1
    rows, cols = img.shape
    for row in range(rows):
        for col in range(cols):
            var = img[row, col]
            if var == 0:
                if var < THRESHOLD:
                    continue
                if var in index_map:
                    index = index_map.index(var)
                    num = index + 1
                else:
                    index = len(index_map)
                    num = index + 1
                    index_map.append(var)
                    points.append([])
                img[row, col] = num
                points[index].append([row, col])
    return img, points

def neighbor_value(img: np.array, offsets, reverse=False):
    rows, cols = img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if not reverse else [rows-1, -1, -1]
    cols_ = [0, cols, 1] if not reverse else [cols-1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if img[row, col] < THRESHOLD:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row + offset[0]), rows-1)
                neighbor_col = min(max(0, col + offset[1]), cols-1)
                neighbor_val = img[neighbor_row, neighbor_col]
                if neighbor_val < THRESHOLD:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            img[row, col] = label
    return img
        
def Two_Pass(img: np.array, neighbor):
    if neighbor == NEIGHBORHOOD_4:
        offsets = OFFSETS_4
    elif neighbor == NEIGHBORHOOD_8:
        offsets = OFFSETS_8
    else:
        raise ValueError("Invalid neighborhood")

    rows, cols = img.shape
    img = neighbor_value(img, offsets)
    img = neighbor_value(img, offsets, reverse=True)
    return img
