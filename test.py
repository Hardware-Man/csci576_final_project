import numpy as np
import cv2
import matplotlib.pyplot as plt
from stitching import Stitcher, AffineStitcher
settings = {
    'detector': 'brisk',
    'crop': False,
    'compensator': 'no',
    'wave_correct_kind': 'no',
    'warper_type': 'spherical',
    'adjuster': 'ray',
}
affine_settings = {
    'detector': 'orb',
    'crop': False,
}
affine_stitcher = AffineStitcher(**affine_settings)
stitcher = Stitcher(**settings)

images = []
images.append(cv2.imread("out/dataset1/video1/keyframes/frame560.png"))
images.append(cv2.imread("out/dataset1/video1/keyframes/frame616.png"))

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

    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)

    plt.imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
    plt.show()
    
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
    return cropExtra(result,0.001)

# result = stitchImages(images[0], images[1])
result = affine_stitcher.stitch(images)
# result = stitcher.stitch(images)
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()

cv2.imwrite("result.png", result)