import cv2
import mediapipe as mp
import time


class PoseDetector:

    def __init__(self, mode = False, upBody = False, smooth=True, detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS

    def getPosition(self, img, draw=True):
        lmList = []
        body_parts = [
            "Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer", "Right Eye Inner", 
            "Right Eye", "Right Eye Outer", "Left Ear", "Right Ear", "Mouth Left", 
            "Mouth Right", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
            "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", "Left Index", 
            "Right Index", "Left Thumb", "Right Thumb", "Left Hip", "Right Hip", 
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Left Heel", 
            "Right Heel", "Left Foot Index", "Right Foot Index"
        ]
        
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy, lm.x, lm.y, body_parts[id]])
                # Add text label for body part name
                if draw and id in list(range(len(body_parts))):
                    cv2.putText(img, body_parts[id], (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                if draw and id in list(range(10)):
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    
        return lmList

    def getBodyPartPositions(self):
        body_part_positions = {}
        body_parts = [
            "Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer", "Right Eye Inner", 
            "Right Eye", "Right Eye Outer", "Left Ear", "Right Ear", "Mouth Left", 
            "Mouth Right", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
            "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", "Left Index", 
            "Right Index", "Left Thumb", "Right Thumb", "Left Hip", "Right Hip", 
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Left Heel", 
            "Right Heel", "Left Foot Index", "Right Foot Index"
        ]
        
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                body_part_positions[body_parts[id]] = (lm.x, lm.y, lm.z)
        
        return body_part_positions