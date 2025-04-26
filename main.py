import numpy as np
import cv2
import os

video_name = 'video_data/video2/real_002.mp4'

def loadFrames(video_path):
    try:
        if not os.path.exists('out/extracted_frames'):
            os.makedirs('out/extracted_frames')
    except OSError:
        print("Error: Creating output directory")

    cap = cv2.VideoCapture(video_name)

    count = 0

    success = 1

    while success:

        success, image = cap.read()

        if success:
            cv2.imwrite("out/extracted_frames/frame%d.jpg" % count, image)
            print(f"Successfully captured frame {count}")

        count += 1

if __name__ == "__main__":
    loadFrames(video_name)