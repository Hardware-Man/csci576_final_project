# imageStitch.py

import numpy as np
import cv2
import glob
import imutils

def print_load_progress(iteration, total, bar_length=50):
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(bar_length * iteration // total)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: [{bar}] {percent}%', end='\r')
    if iteration == total:
        print()

def stitchImage(folder):
	image_paths = glob.glob(folder + '*.jpg')
	images = []
	for i in range(len(image_paths)):
		image = image_paths[i]
		print_load_progress(i+1, len(image_paths))
		img = cv2.imread(image)
		images.append(img)


def processFrames(folders):
	for folder in folders:
		stitchImage(folder)