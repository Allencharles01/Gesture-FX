import streamlit as st
import cv2
import mediapipe as mp
import time
import av # 👈 Essential for the new callback
from streamlit_webrtc import webrtc_streamer

# --- UI Configuration ---
st.set_page_config(layout="centered", page_title="Gesture FX Web")
st.title("🖐️ Gesture FX: Web Edition")

# Initialize Shared State (The Bridge)
if "gesture_trigger" not in st.session_state:
    st.session_state.gesture_trigger = None

# Setup MediaPipe
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# --- The Modern Video Processing Function ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    # Process
    results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            
            # Simple Trigger (Example: Thumb up detected)
            if hand.landmark[4].y < hand.landmark[8].y:
                st.session_state.gesture_trigger = "thumb"

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# THE STREAMER
webrtc_streamer(
    key="gesture-fx",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# --- UI Effect Logic (In Main Thread) ---
if st.session_state.gesture_trigger == "thumb":
    st.balloons()
    st.session_state.gesture_trigger = None # Reset
    st.rerun() # Refresh to show balloons