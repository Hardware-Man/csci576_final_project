import numpy as np
import cv2
import sys
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

video_name = 'video_data/video3/deer1.mp4'
cap = cv2.VideoCapture(video_name)
frame_number = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
success, image1 = cap.read()

if not success:
    sys.exit("Error in reading frame %d" %frame_number)
else:
    # image1 = rotate(image1, 90)
    # cv2.imshow("Frame %d" %frame_number, image1)
    print("Captured frame %d" %frame_number)
    # cv2.waitKey(0)

frame_number = 30
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
success, image2 = cap.read()

if not success:
    sys.exit("Error in reading frame %d" %frame_number)
else:
    # image2 = rotate(image2, 90)
    # cv2.imshow("Frame %d" %frame_number, image2)
    print("Captured frame %d" %frame_number)
    # cv2.waitKey(0)

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

# use orb if sift is not installed
feature_extractor = cv2.SIFT_create()

# find the keypoints and descriptors with chosen feature_extractor
kp1, desc1 = feature_extractor.detectAndCompute(image1, None)
kp2, desc2 = feature_extractor.detectAndCompute(image2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

# Apply ratio test
good_and_second_good_match_list = []
for m in matches:
    if m[0].distance / m[1].distance < 0.5:
        good_and_second_good_match_list.append(m)
good_match_arr = np.asarray(good_and_second_good_match_list)[:, 0]

# show only 30 matches
im_matches = cv2.drawMatchesKnn(
    image1,
    kp1,
    image2,
    kp2,
    good_and_second_good_match_list[0:30],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

# plt.figure(figsize=(20, 20))
# plt.imshow(im_matches)
# plt.title("keypoints matches")
# plt.show()

good_kp_l = np.array([kp1[m.queryIdx].pt for m in good_match_arr])
good_kp_r = np.array([kp2[m.trainIdx].pt for m in good_match_arr])
H, masked = cv2.findHomography(good_kp_r, good_kp_l, cv2.RANSAC, 5.0)

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

result = warpTwoImages(image1, image2, H)
# result = stitchImages(image2, image1)
plt.figure(figsize=(10, 10))
plt.imshow(result)
plt.title("better warping")
plt.show()