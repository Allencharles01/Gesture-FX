import streamlit as st
import cv2
import mediapipe as mp
import time
import av
from streamlit_webrtc import webrtc_streamer

# --- UI Configuration ---
st.set_page_config(layout="centered", page_title="Gesture FX Web")
st.title("🖐️ Hand Triggered Effects")
st.caption("Developed by **Allen Charles** | Hold a gesture for 2 seconds to trigger.")

# Initialize Session States
if 'gesture_trigger' not in st.session_state:
    st.session_state.gesture_trigger = None
if 'last_gesture' not in st.session_state:
    st.session_state.last_gesture = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0
if 'done' not in st.session_state:
    st.session_state.done = False

# Setup MediaPipe 
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def apply_theme(dark_mode=False):
    bg = "#1e1e1e" if dark_mode else "white"
    fg = "white" if dark_mode else "black"
    st.markdown(f"""<style>.stApp {{ background-color: {bg}; color: {fg}; transition: 0.5s; }}</style>""", unsafe_allow_html=True)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    this_gesture = None
    anchor_pt = None

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            pts = hand.landmark
            anchor_pt = (int(pts[0].x * w), int(pts[0].y * h))
            
            t_tip, i_tip, m_tip, r_tip, p_tip = pts[4], pts[8], pts[12], pts[16], pts[20]
            
            # Gesture Logic from your original code
            if t_tip.y < i_tip.y and t_tip.y < m_tip.y:
                this_gesture = "thumb"
            elif i_tip.y < pts[6].y and m_tip.y < pts[10].y and r_tip.y > pts[14].y:
                this_gesture = "peace"
            elif i_tip.y > pts[5].y and m_tip.y > pts[9].y and r_tip.y > pts[13].y and p_tip.y > pts[17].y:
                this_gesture = "fist"
            elif i_tip.y < pts[5].y and m_tip.y < pts[9].y and r_tip.y < pts[13].y and p_tip.y < pts[17].y:
                this_gesture = "palm"

    # Timer Logic inside the video thread
    if this_gesture and this_gesture == st.session_state.last_gesture:
        if not st.session_state.done:
            diff = time.time() - st.session_state.start_time
            if anchor_pt:
                progress = min(1.0, diff / 2.0)
                cv2.circle(img, anchor_pt, 45, (100, 100, 100), 1)
                cv2.ellipse(img, anchor_pt, (45, 45), -90, 0, int(progress * 360), (0, 255, 120), 4)
            
            if diff >= 2.0:
                st.session_state.gesture_trigger = this_gesture
                st.session_state.done = True
    else:
        st.session_state.last_gesture = this_gesture
        st.session_state.start_time = time.time()
        st.session_state.done = False

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the WebRTC Streamer
ctx = webrtc_streamer(
    key="gesture-fx-final",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# Effect Trigger (Main Thread)
if st.session_state.gesture_trigger:
    t = st.session_state.gesture_trigger
    if t == "thumb": st.balloons()
    elif t == "peace": st.snow()
    elif t == "fist": apply_theme(dark_mode=True)
    elif t == "palm": apply_theme(dark_mode=False)
    st.session_state.gesture_trigger = None

# Sidebar Instructions
with st.sidebar:
    st.header("📖 Spellbook")
    st.write("👍 **Thumb**: Balloons")
    st.write("✌️ **Peace**: Snow")
    st.write("✊ **Fist**: Dark Mode")
    st.write("✋ **Palm**: Light Mode")
    st.divider()
    st.info("Portfolio: [allencharles.dev](https://allencharles.dev)")