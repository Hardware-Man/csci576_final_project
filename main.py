import numpy as np
import cv2
import os
import sys

def getVideoPaths():
    if len(sys.argv) != 2:
        print("Usage: python main.py <video number>")
        sys.exit(1)
    
    num = int(sys.argv[1])

    if num not in range(1,6):
        print("Usage: video number must be between 1 and 5")
        sys.exit(1)

    videoPaths = []
    directory = f"video_data/video{num}/"
    if num != 5:
        videoPath = directory + f"real_00{num}.mp4"
        videoPaths.append(videoPath)
        print(f"Queuing {videoPath} for processing")
        return videoPaths
    else:
        forest1 = directory + "forest1.mp4"
        forest2 = directory + "forest2.mp4"
        videoPaths.append(forest1)
        print(f"Queuing {forest1} for processing")
        videoPaths.append(forest2)
        print(f"Queuing {forest2} for processing")
        return videoPaths
        

def loadFrames(videoPaths):
    numVideos = len(videoPaths)
    for i in range(numVideos):
        outputDirectory = f"out/video{i+1}/extracted_frames/"
        try:
            if not os.path.exists(outputDirectory):
                os.makedirs(outputDirectory)
        except OSError:
            print("Error: Creating output directory")

        video_name = videoPaths[i]
        cap = cv2.VideoCapture(video_name)

        count = 0

        success = 1

        while success:
            success, image = cap.read()

            if success:
                cv2.imwrite(outputDirectory + "frame%d.jpg" % count, image)
                print(f"Successfully captured frame {count}")

            count += 1

if __name__ == "__main__":
    videoPaths = getVideoPaths()
    loadFrames(videoPaths)