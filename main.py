import numpy as np
import cv2

video_name = 'video_data/video2/real_002.mp4'
cap = cv2.VideoCapture(video_name)

success, image = cap.read()

if success:
    cv2.imwrite("frame0.jpg", image)
    print("Successfully captured first frame from '" + video_name + "' into 'frame0.jpg'")
else:
    print("Could not read '" + video_name + "'")