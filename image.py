import numpy as np
import cv2
import glob
import os
from imageStitch import *
    
def processFolder(folder):
    image_paths = glob.glob(folder + '*.jpg')
    images = []
    print(f"Reading images in {folder}")
    for i in range(len(image_paths)):
        image = image_paths[i]
        print_load_progress(i+1, len(image_paths))
        img = cv2.imread(image)
        images.append(img)

    return images

def processFrames(folders):
    num = 0
    if len(sys.argv) >= 2:
        arg1 = sys.argv[1]
        if arg1.isnumeric():
            num = int(arg1)
    imageDirectory = f"out/video{num}"
    try:
        if not os.path.exists(imageDirectory):
            os.makedirs(imageDirectory)
    except OSError:
        print("Error: Creating output directory")

    images = []
    for folder in folders:
        images += processFolder(folder)
    try:
        print("Stitching images sequentially with custom stitchImages function")
        final = stitchImageArray(images)
        cv2.imwrite(f"{imageDirectory}/finalImageSequentialStitchImagesFunction.png", final)
        # print("Stitching images with sequential scan using stitching module")
        # final = stitchImageArrayWithModule(images)
        # cv2.imwrite(f"{imageDirectory}/finalImageStitchingModuleSequential.png", final)
        displayImage(final, "Final Image with Sequential Scanning")
    except Exception as e:
        title = f"Sequential stitching failed with exception {e}"
        print(title)
        plt.title(title)
        plt.savefig(f"{imageDirectory}/finalIamgeStitchingModuleSequential.png")
        plt.show()

    try:
        print("Stitching images using divide and conquer")
        finalDC = stitchImageArrayDC(images)
        cv2.imwrite(f"{imageDirectory}/finalImageDC.png", finalDC)
        displayImage(finalDC, "Final Image")
    except Exception as e:
        title = f"Divide and conquer scanning failed with exception {e}"
        print(title)
        plt.title(title)
        plt.savefig(f"{imageDirectory}/finalImageDC.png")
        plt.show()
