import streamlit as st
import cv2
import mediapipe as mp
import time
import av
from streamlit_webrtc import webrtc_streamer

# --- UI Configuration & Branding ---
st.set_page_config(layout="centered", page_title="Gesture FX Web")
st.title("🖐️ Gesture FX: Web Edition")
st.caption("Developed by **Allen Charles** | [allencharles.dev](https://allencharles.dev)")

# Initialize Shared State for the 2-second logic
if "gesture_trigger" not in st.session_state:
    st.session_state.gesture_trigger = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "last_gesture" not in st.session_state:
    st.session_state.last_gesture = None

# Setup MediaPipe
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    current_gesture = None
    
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            pts = hand.landmark
            
            # --- Logic for Gesture Detection ---
            if pts[4].y < pts[8].y and pts[4].y < pts[12].y: # Thumb Up
                current_gesture = "thumb"
            elif pts[8].y < pts[6].y and pts[12].y < pts[10].y: # Peace
                current_gesture = "peace"

    # --- 2 Second Timer Logic ---
    if current_gesture:
        if current_gesture == st.session_state.last_gesture:
            if time.time() - st.session_state.start_time >= 2.0:
                st.session_state.gesture_trigger = current_gesture
        else:
            st.session_state.start_time = time.time()
            st.session_state.last_gesture = current_gesture
    else:
        st.session_state.last_gesture = None

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start Streamer
webrtc_streamer(
    key="gesture-fx",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# --- Effects & Sidebar (The missing parts!) ---
if st.session_state.gesture_trigger:
    if st.session_state.gesture_trigger == "thumb":
        st.balloons()
    elif st.session_state.gesture_trigger == "peace":
        st.snow()
    st.session_state.gesture_trigger = None # Reset

with st.sidebar:
    st.header("📖 Spellbook")
    st.write("👍 **Thumb Up (2s)**: Balloons")
    st.write("✌️ **Peace Sign (2s)**: Snow")
    st.divider()
    st.info("Check out more projects at [allencharles.dev](https://allencharles.dev)")