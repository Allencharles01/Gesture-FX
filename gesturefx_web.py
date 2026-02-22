import streamlit as st
import cv2
import mediapipe as mp
import time
import av
import queue # 👈 Added for thread-safe communication
from streamlit_webrtc import webrtc_streamer

# --- UI Configuration ---
st.set_page_config(layout="centered", page_title="Gesture FX Web")
st.title("🖐️ Hand Triggered Effects")
st.caption("Developed by **Allen Charles**")

# This queue will hold the gesture trigger from the camera thread
result_queue = queue.Queue()

# Setup MediaPipe 
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# We use a simple class to track the 2-second timer inside the thread
class GestureTracker:
    def __init__(self):
        self.last_gesture = None
        self.start_time = 0
        self.done = False

tracker = GestureTracker()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    this_gesture = None
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            pts = hand.landmark
            
            # Tips
            t_tip, i_tip, m_tip, r_tip, p_tip = pts[4], pts[8], pts[12], pts[16], pts[20]
            
            if t_tip.y < i_tip.y and t_tip.y < m_tip.y: this_gesture = "thumb"
            elif i_tip.y < pts[6].y and m_tip.y < pts[10].y: this_gesture = "peace"
            elif i_tip.y > pts[5].y and m_tip.y > pts[9].y and p_tip.y > pts[17].y: this_gesture = "fist"
            elif i_tip.y < pts[5].y and m_tip.y < pts[9].y and p_tip.y < pts[17].y: this_gesture = "palm"

    # Timer Logic (Inside the camera thread)
    if this_gesture and this_gesture == tracker.last_gesture:
        if not tracker.done:
            if time.time() - tracker.start_time >= 2.0:
                result_queue.put(this_gesture) # 👈 Send to the UI thread!
                tracker.done = True
    else:
        tracker.last_gesture = this_gesture
        tracker.start_time = time.time()
        tracker.done = False

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start Streamer
webrtc_streamer(
    key="gesture-fx-final",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# --- THE TRIGGER CHECK (Runs in the Main Thread) ---
try:
    triggered_gesture = result_queue.get_nowait()
    if triggered_gesture == "thumb": st.balloons()
    elif triggered_gesture == "peace": st.snow()
    elif triggered_gesture == "fist": 
        st.markdown("<style>.stApp { background-color: #1e1e1e; color: white; }</style>", unsafe_allow_html=True)
    elif triggered_gesture == "palm": 
        st.markdown("<style>.stApp { background-color: white; color: black; }</style>", unsafe_allow_html=True)
except queue.Empty:
    pass

# Sidebar Instructions
with st.sidebar:
    st.header("📖 Spellbook")
    st.write("👍 **Thumb**: Balloons")
    st.write("✌️ **Peace**: Snow")
    st.write("✊ **Fist**: Dark Mode")
    st.write("✋ **Palm**: Light Mode")
    st.info("Check it out: [allencharles.dev](https://allencharles.dev)")