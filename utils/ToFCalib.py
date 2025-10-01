import h5py
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt

h5_folder = "E:\Projects\ToF\ToF\output"
folder_name = ["50", "50d", "100", "100d", "200", "200d", "300", "300d", "400", "400d"]
# folder_name = "50d"
# The name in folder_name shows the distance and the 'd' means it's in a dark environment
# Followed is plot the comparison of dark and normal environment in one figure
dark_env = {}
normal_env = {}

# for name in folder_name:
#     if "d" in name:
#         dark_env[name] = []
#         files = sorted(os.listdir(f"{h5_folder}/{name}"))
#         for f in tqdm.tqdm(files):
#             fname = os.path.basename(f)
#             h5_fname = f"{h5_folder}/{name}/{f}"
#             h5_file = h5py.File(h5_fname, "r")
#             distance = h5_file["distance"][:]
#             dark_env[name].append(distance)
#             h5_file.close()
#             # print(distance)
#     else:
#         normal_env[name] = []
#         files = sorted(os.listdir(f"{h5_folder}/{name}"))
#         for f in tqdm.tqdm(files):
#             fname = os.path.basename(f)
#             h5_fname = f"{h5_folder}/{name}/{f}"
#             h5_file = h5py.File(h5_fname, "r")
#             distance = h5_file["distance"][:]
#             normal_env[name].append(distance)
#             h5_file.close()
#             # print(distance)
# # plt
# x1 = np.arange(len([name for name in folder_name if "d" in name]))
# x2 = x1 + 0.4

dark_env = []
light_env = []
for name in folder_name:
    files = sorted(os.listdir(f"{h5_folder}/{name}"))
    distances = {"corner": [], "edge": [], "center": [], "all": []}
    for f in tqdm.tqdm(files):
        fname = os.path.basename(f)
        h5_fname = f"{h5_folder}/{name}/{f}"
        h5_file = h5py.File(h5_fname, "r")
        distance = h5_file["distance"][:]
        distances["all"].append(distance)
        h5_file.close()
        print(distance)

    distances["all"] = np.array(distances["all"])
    print(distances["all"].shape)
    sum = 0
    valid_zones = 0

    for dist in distances["all"]:
        print(dist)
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                if dist[i, j] > 1.5 and dist[i, j] < 450:
                    sum += dist[i, j]
                    valid_zones += 1
                    if i in [0, 1, 6, 7] and j in [0, 1, 6, 7]:
                        distances["corner"].append(dist[i, j])
                    elif i in [0, 1, 6, 7] or j in [0, 1, 6, 7]:
                        distances["edge"].append(dist[i, j])
                    else:
                        distances["center"].append(dist[i, j])
                else:
                    continue
    mean_distance = sum / valid_zones
    if "d" in name:
        dark_env.append(mean_distance)
    else:
        light_env.append(mean_distance)
    print(f"Average distance: {sum/valid_zones}, {valid_zones}")
    print(
        f"Average corner distance: {np.mean(distances['corner'])}, {len(distances['corner'])}"
    )
    print(
        f"Average edge distance: {np.mean(distances['edge'])}, {len(distances['edge'])}"
    )
    print(
        f"Average center distance: {np.mean(distances['center'])}, {len(distances['center'])}"
    )

    ## plot the distribution of corner, edge, center separately in different x-axis
    plt.figure()
    plt.boxplot(
        [distances["corner"], distances["edge"], distances["center"]],
        labels=["corner", "edge", "center"],
        # notch=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightcyan"),
        meanline=True,
        meanprops=dict(color="blue"),
        showmeans=True,
        medianprops=dict(color="lightcyan"),
        showfliers=False,
    )
    plt.title(f"Distance distribution -- {name}mm")
    plt.text(1.2, mean_distance + 2, s="%.3f" % np.mean(distances["corner"]))
    plt.text(2.2, mean_distance, s="%.3f" % np.mean(distances["edge"]))
    plt.text(3.2, mean_distance - 2, s="%.3f" % np.mean(distances["center"]))
    # plot the average distance
    plt.axhline(
        y=mean_distance,
        color="r",
        linestyle="--",
        label=f"Avg distance:{mean_distance}",
    )
    plt.legend()
    plt.show()
    plt.close()
species = ("50", "100", "200", "300", "400")
x = np.arange(len(species))
width = 0.35
fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, dark_env, width, label="Dark Environment")
rects2 = ax.bar(x + width / 2, light_env, width, label="Light Environment")
ax.set_ylabel("Distance (mm)")
ax.set_title("Average distance in different environments")
ax.set_xticks(x)
ax.set_xticklabels(species)
ax.legend()
plt.show()
