## Read images from '/output' and save as a video
import cv2
import os
img_dir = './kmeans'
img_files = os.listdir(img_dir)
video_name = 'kmeans.avi'
fps = 2
frame_size = (640,640)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

for img_file in img_files:
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    video.write(img)
    print(f'Writing {img_file} to video')

video.release()