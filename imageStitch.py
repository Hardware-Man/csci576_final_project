import numpy as np
import cv2
import sys
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

def print_load_progress(iteration, total, bar_length=50):
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(bar_length * iteration // total)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: [{bar}] {percent}%', end='\r')
    if iteration == total:
        print()

def stitchImages(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use BFMatcher to find matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    start = (abs(int(dst_pts[0][0][0]-src_pts[0][0][0])), abs(int(dst_pts[0][0][1]-src_pts[0][0][1])))
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((start[1]+h1,start[0]+w1,3), np.uint8)
    vis[start[1]:start[1]+h1, start[0]:start[0]+w1, :3] = img1
    vis[:h2, :w2, :3] = img2
    return vis

def featureExtract(img1, img2):
    feature_extractor = cv2.SIFT_create()

    # find the keypoints and descriptors with chosen feature_extractor
    kp1, desc1 = feature_extractor.detectAndCompute(img1, None)
    kp2, desc2 = feature_extractor.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    good_and_second_good_match_list = []
    for m in matches:
        if m[0].distance / m[1].distance < 0.5:
            good_and_second_good_match_list.append(m)
    good_match_arr = np.asarray(good_and_second_good_match_list)[:, 0]

    good_kp_l = np.array([kp1[m.queryIdx].pt for m in good_match_arr])
    good_kp_r = np.array([kp2[m.trainIdx].pt for m in good_match_arr])
    H, masked = cv2.findHomography(good_kp_r, good_kp_l, cv2.RANSAC, 5.0)
    return H, masked

def warpTwoImages(img1, img2, H):
    """warp img2 to img1 with homograph H
    from: https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
    result[t[1] : h1 + t[1], t[0] : w1 + t[0]] = img1
    return result

def stitchImageArray(images):
    print(f"Stitching {len(images)} images together")
    out = images[0]
    if (len(images) == 1):
        return out
    for i in range(1, len(images)):
	    print_load_progress(i, len(images))
	    H, masked = featureExtract(out, images[i])
	    out = warpTwoImages(out, images[i], H)
    return out

def displayImage(img, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(title)
    plt.show()