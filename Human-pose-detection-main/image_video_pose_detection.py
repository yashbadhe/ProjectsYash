import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import tempfile

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.title("Pose Estimation and Activity Recognition with Adjustable Thresholds")
st.sidebar.title("Upload an Image or a Video")
file = st.sidebar.file_uploader("Choose a file", type=["jpg", "png", "mp4"])

st.sidebar.header("Threshold Adjustments")
standing_threshold = st.sidebar.slider("Standing Threshold", 0.05, 0.5, 0.1, 0.01)
sitting_threshold = st.sidebar.slider("Sitting Threshold", 0.05, 0.5, 0.1, 0.01)
lying_down_threshold = st.sidebar.slider("Lying Down Threshold", 0.01, 0.2, 0.05, 0.01)

# classify pose 
def classify_pose(landmarks, standing_th, sitting_th, lying_th):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

    shoulder_avg_height = (left_shoulder.y + right_shoulder.y) / 2
    hip_avg_height = (left_hip.y + right_hip.y) / 2
    knee_avg_height = (left_knee.y + right_knee.y) / 2

    if shoulder_avg_height < hip_avg_height - standing_th and hip_avg_height < knee_avg_height:
        return "Standing"
    elif hip_avg_height < knee_avg_height - sitting_th and abs(shoulder_avg_height - hip_avg_height) < sitting_th:
        return "Sitting"
    elif abs(shoulder_avg_height - hip_avg_height) < lying_th:
        return "Lying Down"
    else:
        return "Unknown Pose"

# process image
def process_image(image, standing_th, sitting_th, lying_th):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
            )
            pose_label = classify_pose(results.pose_landmarks.landmark, standing_th, sitting_th, lying_th)
            return image, pose_label
        return image, None

#  process video
def process_video(video_path, standing_th, sitting_th, lying_th):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
                )
                pose_label = classify_pose(results.pose_landmarks.landmark, standing_th, sitting_th, lying_th)
                cv2.putText(frame, pose_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            stframe.image(frame, channels='BGR')
        cap.release()

if file is not None:
    if file.type.startswith("image"):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_image, pose_label = process_image(image, standing_threshold, sitting_threshold, lying_down_threshold)
        st.image(processed_image, caption=f"Processed Image - Pose: {pose_label}", use_column_width=True)
    elif file.type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(file.read())
        process_video(tfile.name, standing_threshold, sitting_threshold, lying_down_threshold)
else:
    st.write("Please upload an image or a video file.")


