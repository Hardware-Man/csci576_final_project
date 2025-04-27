import numpy as np
import cv2
import os
import sys
from image import *

def getVideoPaths():
    if len(sys.argv) != 2:
        print("Usage: python main.py <video number>")
        sys.exit(1)

    if num not in range(1,6):
        print("Usage: video number must be between 1 and 5")
        sys.exit(1)

    videoPaths = []
    directory = f"video_data/video{num}/"
    if num != 5 and num != 3:
        videoPath = directory + f"real_00{num}.mp4"
        videoPaths.append(videoPath)
        print(f"Queuing {videoPath} for processing")
        return videoPaths
    elif num == 5:
        forest1 = directory + "forest1.mp4"
        forest2 = directory + "forest2.mp4"
        videoPaths.append(forest1)
        print(f"Queuing {forest1} for processing")
        videoPaths.append(forest2)
        print(f"Queuing {forest2} for processing")
        return videoPaths
    else:
        deer1 = directory + "deer1.mp4"
        deer2 = directory + "deer2.mp4"
        videoPaths.append(deer1)
        print(f"Queuing {deer1} for processing")
        videoPaths.append(deer2)
        print(f"Queuing {deer2} for processing")
        return videoPaths
        

def loadFrames(videoPaths):
    numVideos = len(videoPaths)
    outputDirectories = [] # stores the folders that contain the frames for each video

    for i in range(numVideos):
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

        video_name = videoPaths[i]
        cap = cv2.VideoCapture(video_name)

        if cap.isOpened():
            count = 0

            success = 1

            tick = 0

            while success:
                success, image = cap.read()

                tick += 1
                if tick == 3:
                    tick = 0
                if tick == 1:
                    if success:
                        cv2.imwrite(outputDirectory + f"frame{str(count).zfill(3)}.jpg", image)

                    count += 1
            print(f"Captured {count} frames")
            cap.release()

    return outputDirectories

if __name__ == "__main__":
    num = int(sys.argv[1])
    videoPaths = getVideoPaths()
    frameFolders = loadFrames(videoPaths)
    processFrames(frameFolders)