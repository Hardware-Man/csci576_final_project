# image.py

import numpy as np
import cv2
import glob
from imageStitch import *
	
def processFolder(folder, imageNum):

	image_paths = glob.glob(folder + '*.jpg')
	images = []
	for i in range(len(image_paths)):
		image = image_paths[i]
		print_load_progress(i+1, len(image_paths))
		img = cv2.imread(image)
		images.append(img)

	out = stitchImageArray(images)
	cv2.imwrite(f"out/stitchedImage{imageNum}.png", out)
	displayImage(out, f"Stitched Image #{imageNum+1}")
	return images

def processFrames(folders):
	count = 0
	stitches = []
	for folder in folders:
		stitches += processFolder(folder, count)
		count += 1
	
	final = stitchImageArray(stitches)
	cv2.imwrite("out/finalImage.png", final)
	displayImage(final, "Final Image")
