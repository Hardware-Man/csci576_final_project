import cv2
import matplotlib.pyplot as plt
import sys
from stitching import Stitcher, AffineStitcher

def createStitcher(adjuster='ray'):
    warp = 'spherical'
    if len(sys.argv) > 3:
        warp = sys.argv[3]
    settings = {
        'detector': 'sift',
        'crop': False,
        "try_use_gpu": True,
        'wave_correct_kind': 'no',
        'warper_type': warp,
        'adjuster': adjuster,
    }
    print(f"Creating stitcher with '{warp}' warper and '{adjuster}' adjuster")
    return Stitcher(**settings)

def createAffineStitcher():
    affine_settings = {
        'detector': 'orb',
        'crop': False,
        "try_use_gpu": True,
        "compensator": "gain_blocks",
    }
    print(f"Creating affine stitcher")
    return AffineStitcher(**affine_settings)

def print_load_progress(iteration, total, bar_length=50):
    percent = "{0:.1f}".format(min(100, 100 * (iteration / float(total))))
    filled_length = int(bar_length * iteration // total)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: [{bar}] {percent}%', end='\r')
    if iteration == total:
        print()

def displayImage(img, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()