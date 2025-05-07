import os
import sys
import shutil
import subprocess
import shlex
from image import *

def getVideoPaths():
    if len(sys.argv) < 2:
        print("Usage: python main.py <dataset number|video path> <stitch mode> <warper type>")
        sys.exit(1)

    arg1 = sys.argv[1]
    if arg1.isnumeric():
        dataset_num = int(arg1)
        if dataset_num not in range(1,6):
            print("Usage: video number must be between 1 and 5")
            sys.exit(1)

        videoPaths = []
        directory = f"video_data/video{dataset_num}/"
        if dataset_num == 1 or dataset_num == 2 or dataset_num == 4:
            videoPath = directory + f"real_00{dataset_num}.mp4"
            videoPaths.append(videoPath)
            print(f"Queuing {videoPath} for processing")
        elif dataset_num == 5:
            forest1 = directory + "forest1.mp4"
            forest2 = directory + "forest2.mp4"
            videoPaths.append(forest1)
            print(f"Queuing {forest1} for processing")
            videoPaths.append(forest2)
            print(f"Queuing {forest2} for processing")
        elif dataset_num == 3:
            deer1 = directory + "deer1.mp4"
            deer2 = directory + "deer2.mp4"
            videoPaths.append(deer1)
            print(f"Queuing {deer1} for processing")
            videoPaths.append(deer2)
            print(f"Queuing {deer2} for processing")
    else:
        videoFolder = arg1
        videoFolder = videoFolder.replace("\\", "/")
        if videoFolder[-1] != "/":
            videoFolder += "/"
        dataset_num = 0
        videoPaths = []
        for file in os.listdir(videoFolder):
            if file.endswith('.mp4'):
                videoPath = videoFolder + file
                videoPaths.append(videoPath)
                print(f"Queuing {videoPath} for processing")
        
    return dataset_num, videoPaths

def loadFrames(dataset_num, videoPaths):
    numVideos = len(videoPaths)
    outputDirectories = [] # stores the folders that contain the frames for each video

    for i in range(numVideos):
        video_name = videoPaths[i]
        outputDirectory = f"out/dataset{dataset_num}/video{i+1}/keyframes/"
        outputDirectories.append(outputDirectory)
        try:
            if not os.path.exists(outputDirectory):
                os.makedirs(outputDirectory)
            else:
                if dataset_num != 0:
                    print(f"Already loaded frames for dataset {dataset_num} video {i+1}")
                    continue
                else:
                    shutil.rmtree(outputDirectory)
                    os.makedirs(outputDirectory)
        except OSError:
            print("Error: Creating output directory")

        cmd_string = f"ffmpeg -loglevel error -skip_frame nokey -i {video_name} -vsync vfr -frame_pts true {outputDirectory}frame%03d.png"

        print("running '" + cmd_string + "'")

        result = subprocess.run(shlex.split(cmd_string))

        if result.returncode == 0:
            print(f"Captured keyframes for {video_name} into {outputDirectory}")
        else:
            print(f"Error occured when capturing keyframes for {video_name}")

        # cap = cv2.VideoCapture(video_name)

        # if cap.isOpened():
        #     success = 1
        #     frame_number = 0
        #     count = 0

        #     while success:
        #         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        #         success, image = cap.read()

        #         if success:
        #             cv2.imwrite(outputDirectory + f"frame{frame_number:03}.jpg", image)
        #             frame_number += 15
        #             count += 1

        #     print(f"Captured {count} frames")
        #     cap.release()

    return outputDirectories

if __name__ == "__main__":
    try:
        dataset_num, videoPaths = getVideoPaths()
        frameFolders = loadFrames(dataset_num, videoPaths)
        if len(sys.argv) > 2:
            arg2 = sys.argv[2]
            if arg2.isnumeric():
                mode = int(arg2)
                if mode not in range(0,3):
                    print("Usage: mode id value must be between 0 and 2")
                    sys.exit(1)
                processFrames(frameFolders, mode)
            else:
                print("Usage: mode id must be an integer between 0 and 2")
                sys.exit(1)
        else:
            processFrames(frameFolders)
    except KeyboardInterrupt:
        print('^C')
        sys.exit()