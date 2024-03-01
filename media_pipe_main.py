
import cv2
import time
import pose_estimation_class as pm
import mediapipe as mp
import argparse

import numpy as np

# https://medium.com/nerd-for-tech/deep-learning-based-human-pose-estimation-using-opencv-and-mediapipe-d0be7a834076

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
ap.add_argument("-o", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--fps", type=int, default=30,
	help="set fps of output video")
ap.add_argument("-b", "--black", type=str, default=False,
	help="set black background")
args = vars(ap.parse_args())


pTime = 0
# black_flag = eval(args["black"])
black_flag = False
cap = cv2.VideoCapture(args["input"])
out = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc(*"MJPG"), 
                      args["fps"], (int(cap.get(3)), int(cap.get(4))))

detector = pm.PoseDetector()

while(cap.isOpened()):
    success, img = cap.read()
    
    if success == False:
        break
    
    img, p_landmarks, p_connections = detector.findPose(img, False)
    
    # use black background
    if black_flag:
        img = img * 0
    
    # draw points
    mp.solutions.drawing_utils.draw_landmarks(img, p_landmarks, p_connections)
    lmList = detector.getPosition(img)

    # TODO is it always in this order?
    head_positions = lmList[:10]
    # import pdb;pdb.set_trace()
    # head_position_avg = np.array([np.array([p.x, p.y, p.z]) for p in head_positions])
    head_position_avg = np.array([np.array([p[3], p[4]]) for p in head_positions]).mean(axis=0)
    h, w, c = img.shape
    # head_position_avg_img = (int(head_position_avg[0] * w), int(head_position_avg[1] * h))
    
    # cv2.circle(img, head_position_avg_img, 10, (255, 255, 0))

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    out.write(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()
