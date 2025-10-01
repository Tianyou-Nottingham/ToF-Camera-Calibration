import numpy as np  
from TOF_RANSAC import Plane
import configs.config as cfg

## K-means clustering
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
    centers = [np.mean(data[0], axis=0), np.mean(data[1], axis=0)]
    for i in range(len(data)):
        for point in data[i]:
            if np.linalg.norm(np.array(point) - centers[i]) > threshold:
                data[i].remove(point)
    return data

def plane_kmeans(data, k = 2):
    ## plane_kmeans should cluster the points depending on the plane they belong to
    ## data:[8, 8] depth map
    ## k: number of clusters
    ## return the segmentation groups
    ## For the zones divided by the segmentation line, we calculate the mean depth value. And set the new 
    w, h = data.shape
    ## random initialization
    centers = [np.random.choice(w, k), np.random.choice(h, k)]
    plane = []
    for i in range(k):
        plane.append(Plane(np.array([pow(-1, i), 0, 1]), data.min() + (i+1)/(k+1) * (data.max() - data.min())))
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
                for m in range(k):
                    ## Calculate the depth distance for every zone
                    dist.append(plane[m].solve_distance(np.array([i, j, data[i, j]])))

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

        value = {}
        for idx in range(k):
            if cluster_index[idx] == []:
                cluster_index[idx] = [(0, 0)]
                centers[idx] = [4, 4]
            else:
                centers[idx] = np.mean(cluster_index[idx], axis=0)
            value.update({np.mean(cluster_value[idx], axis=0): cluster_index[idx]})
            ## Update the plane   
        for m in range(k):
            plane[m].ToF_RANSAC(np.array([[i, j, data[i, j]] for [i, j] in cluster_index[m]]))
        value = sorted(value.items(), key=lambda x: x[0])
    return list(dict(value).values()) ## return the cluster index