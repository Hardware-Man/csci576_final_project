import cv2
import glob
import os
from imageStitch import *

def processFolder(folder):
    image_paths = glob.glob(folder + '*.png')
    images = []
    print(f"Reading images in {folder}")
    for i in range(len(image_paths)):
        image = image_paths[i]
        print_load_progress(i+1, len(image_paths))
        img = cv2.imread(image)
        images.append(img)

    return images

def processFrames(folders, mode=0):
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

    if mode == 0:
        stitcher = createAffineStitcher()
    elif mode == 1:
        stitcher = createStitcher()
    elif mode == 2:
        stitcher = createStitcher('reproj')

    print(f"Stitching images")

    try:
        final = stitcher.stitch(images)
        cv2.imwrite(f"{imageDirectory}/final_image_{stitcher.settings['warper_type']}_{stitcher.settings['adjuster']}.png", final)
        displayImage(final, f"Final Image with {str(stitcher.settings['warper_type']).capitalize()} Warper and {str(stitcher.settings['adjuster']).capitalize()} Adjuster")
    except Exception as e:
        title = f"Stitching failed with exception {e}"
        print(title)
        plt.title(title)
        plt.savefig(f"{imageDirectory}/final_image_{stitcher.settings['warper_type']}_{stitcher.settings['adjuster']}.png")
        plt.show()