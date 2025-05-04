import numpy as np
import cv2
import os
import sys
import glob
from image import *

def getVideoPaths():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video number|video path> <warp mode>")
        sys.exit(1)

    arg1 = sys.argv[1]
    if arg1.isnumeric():
        num = int(arg1)
        if num not in range(1,6):
            print("Usage: video number must be between 1 and 5")
            sys.exit(1)

        videoPaths = []
        directory = f"video_data/video{num}/"
        if num != 5 and num != 3:
            videoPath = directory + f"real_00{num}.mp4"
            videoPaths.append(videoPath)
            print(f"Queuing {videoPath} for processing")
        elif num == 5:
            forest1 = directory + "forest1.mp4"
            forest2 = directory + "forest2.mp4"
            videoPaths.append(forest1)
            print(f"Queuing {forest1} for processing")
            videoPaths.append(forest2)
            print(f"Queuing {forest2} for processing")
        else:
            deer1 = directory + "deer1.mp4"
            deer2 = directory + "deer2.mp4"
            videoPaths.append(deer1)
            print(f"Queuing {deer1} for processing")
            videoPaths.append(deer2)
            print(f"Queuing {deer2} for processing")
        return num, videoPaths
    else:
        videoFolder = arg1
        num = int(videoFolder[-1])
        videoPaths = []
        for files in os.listdir(videoFolder):
            if files.endswith('.mp4'):
                videoPaths.append(files)
                print(f"Queuing {files} for processing")
        return num, videoPaths

def loadFrames(videoPaths):
    numVideos = len(videoPaths)
    outputDirectories = [] # stores the folders that contain the frames for each video

    for i in range(numVideos):
        video_name = videoPaths[i]
        outputDirectory = f"out/dataset{num}/video{i+1}/extracted_frames/"
        outputDirectories.append(outputDirectory)
        try:
            if not os.path.exists(outputDirectory):
                os.makedirs(outputDirectory)
            else:
                print(f"Already loaded frames for dataset {num} video {i+1}")
                continue
        except OSError:
            print("Error: Creating output directory")

        cap = cv2.VideoCapture(video_name)

        if cap.isOpened():
            success = 1
            frame_number = 0
            count = 0

            while success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                success, image = cap.read()

                if success:
                    cv2.imwrite(outputDirectory + f"frame{frame_number:03}.jpg", image)
                    frame_number += 15
                    count += 1

            print(f"Captured {count} frames")
            cap.release()

    return outputDirectories

if __name__ == "__main__":
    try:
        num, videoPaths = getVideoPaths()
        frameFolders = loadFrames(videoPaths)
        processFrames(frameFolders)
    except KeyboardInterrupt:
        print('^C')
        sys.exit()