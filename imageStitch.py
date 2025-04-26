# imageStitch.py

import numpy as np
import cv2
import glob
import imutils

def stitchImage(folder):
	image_paths = glob.glob(folder + '*.jpg')
	images = []
	for image in image_paths:
		img = cv2.imread(image)
		images.append(img)
	cv2.imshow("image", images[0])
	cv2.waitKey(0)

def processFrames(folders):
	for folder in folders:
		stitchImage(folder)