import numpy as np
import cv2
import glob
from imageStitch import *
    
def processFolder(folder):
    image_paths = glob.glob(folder + '*.jpg')
    images = []
    for i in range(len(image_paths)):
        image = image_paths[i]
        print_load_progress(i+1, len(image_paths))
        img = cv2.imread(image)
        images.append(img)

    return images

def processFrames(folders):
    images = []
    for folder in folders:
        images += processFolder(folder)
    
    final = stitchImageArray(images)
    cv2.imwrite("out/finalImage.png", final)
    displayImage(final, "Final Image")
