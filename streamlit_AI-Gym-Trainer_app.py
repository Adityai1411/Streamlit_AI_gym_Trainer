import streamlit as st
import cv2
import mediapipe as mp
import math
import tempfile
from PIL import Image

st.set_page_config(page_title="Full Body AI Gym Trainer", layout="wide")

# ==========================
# Core Classes
# ==========================

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    def estimate(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        return results.pose_landmarks

class RepCounter:
    def __init__(self):
        self.count = 0
        self.stage = None

    def calculate_angle(self, a, b, c):
        radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
        angle = abs(radians * 180.0 / math.pi)
        if angle > 180:
            angle = 360 - angle
        return angle

    def update(self, landmarks):
        if landmarks is None:
            return self.count
        # Left leg
        l_hip = [landmarks.landmark[23].x, landmarks.landmark[23].y]
        l_knee = [landmarks.landmark[25].x, landmarks.landmark[25].y]
        l_ankle = [landmarks.landmark[27].x, landmarks.landmark[27].y]
        angle_left = self.calculate_angle(l_hip, l_knee, l_ankle)
        # Right leg
        r_hip = [landmarks.landmark[24].x, landmarks.landmark[24].y]
        r_knee = [landmarks.landmark[26].x, landmarks.landmark[26].y]
        r_ankle = [landmarks.landmark[28].x, landmarks.landmark[28].y]
        angle_right = self.calculate_angle(r_hip, r_knee, r_ankle)
        avg_knee_angle = (angle_left + angle_right) / 2
        # Stage detection
        if avg_knee_angle < 90:
            self.stage = "down"
        if avg_knee_angle > 160 and self.stage == "down":
            self.stage = "up"
            self.count += 1
        return self.count

class FormChecker:
    def check(self, landmarks):
        if landmarks is None:
            return "No person detected"
        l_shoulder = [landmarks.landmark[11].x, landmarks.landmark[11].y]
        r_shoulder = [landmarks.landmark[12].x, landmarks.landmark[12].y]
        l_hip = [landmarks.landmark[23].x, landmarks.landmark[23].y]
        r_hip = [landmarks.landmark[24].x, landmarks.landmark[24].y]
        torso_slope = abs((l_shoulder[1]+r_shoulder[1])/2 - (l_hip[1]+r_hip[1])/2)
        if torso_slope < 0.05:
            return "Back too bent"
        return "Good form"

# ==========================
# App UI
# ==========================

st.title("🏋️‍♂️ Full Body AI Gym Trainer")
st.write("Upload a video to track exercises, count reps, and get form feedback.")

exercise = st.selectbox("Select Exercise", ["Squat", "Push-Up", "Lunge"])
uploaded_file = st.file_uploader("Upload Exercise Video", type=["mp4", "mov", "avi"])

# Initialize modules
pose = PoseEstimator()
counter = RepCounter()
form = FormChecker()

if uploaded_file is not None:
    # Save video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        landmarks = pose.estimate(frame)
        reps = counter.update(landmarks)
        feedback = form.check(landmarks)
        # Overlay info
        cv2.putText(frame, f"Exercise: {exercise}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Reps: {reps}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Form: {feedback}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        # Convert to RGB and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)
