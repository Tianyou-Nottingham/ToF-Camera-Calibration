import matplotlib.pyplot as plt
import cv2
import os
import time
import pyrealsense2 as rs
import numpy as np

def save_frames_from_bag(bag_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_file)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)

    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            timestamp = int(time.time() * 1000)
            color_filename = os.path.join(output_dir, f"color_{timestamp}.png")
            depth_filename = os.path.join(output_dir, f"depth_{timestamp}.png")

            cv2.imwrite(color_filename, color_image)
            cv2.imwrite(depth_filename, depth_image)

            print(f"Saved {color_filename} and {depth_filename}")

    except KeyboardInterrupt:
        print("Stopped saving frames")

    finally:
        pipeline.stop()

bag_file = r'C:\Users\ezxtz6\Documents\20241204_175613.bag'
output_dir = r'C:\Users\ezxtz6\OneDrive - The University of Nottingham\Pictures\RealSense'
save_frames_from_bag(bag_file, output_dir)


