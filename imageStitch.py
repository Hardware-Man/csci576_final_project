import numpy as np
import cv2
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import sys
from stitching import Stitcher, AffineStitcher

'''
settings = {
    'detector': 'orb',
    'crop': False,
    'confidence_threshold': 0.2,
    # 'warper_type': 'spherical',
    # 'warper_type': 'plane',
    'warper_type': 'compressedPlaneA2B1',
    # 'warper_type': 'paniniA2B1',
    'compensator': 'no',
    'blender_type': "no",
}
'''

def createStitcher():
    warp = 'plane'
    if len(sys.argv) == 3:
        warp = sys.argv[2]
    settings = {
        'detector': 'orb',
        'crop': False,
        'confidence_threshold': 0.2,
        'warper_type': warp,
        'compensator': 'no',
        'blender_type': "no",
    }
    
    affine_settings = {
        'detector': 'orb',
        'crop': False,
        'confidence_threshold': 0.2,
        'compensator': 'no',
        'blender_type': "no",
    }
    print(f"Creating stitcher with warper_type: {warp}")
    return Stitcher(**settings), AffineStitcher(**affine_settings)

def print_load_progress(iteration, total, bar_length=50):
    percent = "{0:.1f}".format(min(100, 100 * (iteration / float(total))))
    filled_length = int(bar_length * iteration // total)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: [{bar}] {percent}%', end='\r')
    if iteration == total:
        print()

feature_extractor = cv2.ORB.create(nfeatures=1500, fastThreshold=3)

def averageTuple(tupleList):
    avgX, avgY = 0,0
    for tuple in tupleList:
        avgX += tuple[0]
        avgY += tuple[1]
    return (int(avgX/len(tupleList)),int(avgY/len(tupleList)))

def tupleInRange(t1, t2, dif=3):
    if t1[0] + dif > t2[0] and t1[0] - dif < t2[0]:
        if t1[1] + dif > t2[1] and t1[1] - dif < t2[1]:
            return True
    return False

def cropExtra(img,extraRange=0.0):
    y, x = img[:, :, 2].nonzero()
    minx = int(np.min(x)*(1-extraRange))
    miny = int(np.min(y)*(1-extraRange))
    maxx = int(np.max(x)*(1+extraRange))
    maxy = int(np.max(y)*(1+extraRange))
    return img[miny:maxy, minx:maxx]

def stitchImages(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = feature_extractor.detectAndCompute(gray1,None)
    kp2, des2 = feature_extractor.detectAndCompute(gray2,None)

    # img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
    # img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)

    # plt.imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
    # plt.show()
    # plt.imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
    # plt.show()
    
    # Brute Force Matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    good_matches = matches[:10]
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    pointsList = []
    for index in range(0,len(src_pts)):
        curPoint = (int(dst_pts[index][0][0]-src_pts[index][0][0])), (int(dst_pts[index][0][1]-src_pts[index][0][1]))
        pointsList.append(curPoint)

    start = pointsList[0]
    avgTuple = averageTuple(pointsList)
    if not tupleInRange(start, avgTuple): return img1

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    ax = abs(start[0])
    ay = abs(start[1])
    result = np.zeros((ay+h1,ax+w1,3), np.uint8)

    ofst2 = (ax if start[0]<0 else 0, ay if start[1]<0 else 0)
    ofst1 = (0 if start[0]<0 else ax, 0 if start[1]<0 else ay)
    result[ofst1[1]:ofst1[1]+h1, ofst1[0]:ofst1[0]+w1, :3] = img1
    result[ofst2[1]:ofst2[1]+h2, ofst2[0]:ofst2[0]+w2, :3] = img2
    return cropExtra(result, 0.001)

def stitchImageArray(images):
    print(f"Stitching {len(images)} images together")
    out = images[0]
    if (len(images) == 1):
        return out
    
    for i in range(1, len(images)):
        print_load_progress(i, len(images))
        out = stitchImages(out, images[i])
        # out = stitcher.stitch([out, images[i]])

    return out

def stitchImageArrayWithModule(images):
    stitcher, affine_stitcher = createStitcher()
    print(f"Stitching {len(images)} images together")
    out = images[0]
    if (len(images) == 1):
        return out
    
    for i in range(1, len(images)):
        print_load_progress(i, len(images))
        out = stitcher.stitch([out, images[i]])

    return out

def stitchImageArrayDC(images):
    backupImages = images
    stitcher, affine_stitcher = createStitcher()
    level = 0
    while len(images) > 1:
        print(f"\nlevel {level}")
        print(f"stitching {len(images)} images")
        new_images = []
        for i in range(0, len(images)-1, 2):
            try:
                print_load_progress(i, len(images)-1)
                if level == 0:
                    image = affine_stitcher.stitch([images[i], images[i+1]])
                    new_images.append(image)
                else:
                    image = stitcher.stitch([images[i], images[i+1]])
                    new_images.append(image)
                print_load_progress(i+2, len(images)-1)
            except Exception as e:
                print(f"\nError {e}")
                print(f"Falling back to stitching with stitchImages")
                try:
                    new_images.append(stitchImages(images[i], images[i+1]))
                except Exception as e:
                    print(f"Error {e}")
                    print(f"stitchImages failed. Restarting with stitchImageArray")
                    return stitchImageArray(backupImages)
        if len(images) % 2 == 1:
            new_images.append(images[len(images)-1])
        images = new_images
        level += 1
    return images[0]

    

def displayImage(img, title, level = 0, stitch = 0):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()