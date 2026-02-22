import streamlit as st
import cv2
import mediapipe as mp
import time
import av
import queue
from streamlit_webrtc import webrtc_streamer

# --- 1. Branding & UI ---
st.set_page_config(layout="centered", page_title="Gesture FX Web")
st.title("🖐️ Hand Triggered Effects")
st.caption("Developed by **Allen Charles** | [allencharles.dev](https://allencharles.dev)")

# --- 2. The Communication Bridge ---
# Using a shared queue to pass messages from camera to UI
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

# --- 3. Setup MediaPipe ---
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Internal timer state
class Tracker:
    last_gesture = None
    start_time = 0
    done = False

gesture_tracker = Tracker()

# --- 4. Video Engine (Drawing & Logic) ---
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
            if t_tip.y < i_tip.y and t_tip.y < m_tip.y: this_gesture = "thumb"
            elif i_tip.y < pts[6].y and m_tip.y < pts[10].y: this_gesture = "peace"
            elif i_tip.y > pts[5].y and m_tip.y > pts[9].y: this_gesture = "fist"
            elif i_tip.y < pts[5].y and m_tip.y < pts[9].y: this_gesture = "palm"

    # Timer Logic + Drawing Progress Ring
    if this_gesture and this_gesture == gesture_tracker.last_gesture:
        if not gesture_tracker.done:
            diff = time.time() - gesture_tracker.start_time
            if anchor_pt:
                progress = min(1.0, diff / 2.0)
                cv2.circle(img, anchor_pt, 45, (100, 100, 100), 1)
                cv2.ellipse(img, anchor_pt, (45, 45), -90, 0, int(progress * 360), (0, 255, 120), 4)
            
            if diff >= 2.0:
                st.session_state.result_queue.put(this_gesture)
                gesture_tracker.done = True
    else:
        gesture_tracker.last_gesture = this_gesture
        gesture_tracker.start_time = time.time()
        gesture_tracker.done = False

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. Start Camera ---
ctx = webrtc_streamer(
    key="gesture-fx",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# --- 6. THE "LIVE LISTENER" ---
# This loop forces Streamlit to check the queue constantly while the camera is on
status_placeholder = st.empty()

while ctx.state.playing:
    try:
        result = st.session_state.result_queue.get_nowait()
        if result == "thumb": st.balloons()
        elif result == "peace": st.snow()
        elif result == "fist": 
            st.markdown("<style>.stApp { background-color: #1e1e1e; color: white; transition: 0.5s; }</style>", unsafe_allow_html=True)
        elif result == "palm": 
            st.markdown("<style>.stApp { background-color: white; color: black; transition: 0.5s; }</style>", unsafe_allow_html=True)
    except queue.Empty:
        time.sleep(0.1) # Check 10 times per second
        continue

# --- 7. Sidebar ---
with st.sidebar:
    st.header("📖 Spellbook")
    st.write("👍 **Thumb (2s)**: Balloons")
    st.write("✌️ **Peace (2s)**: Snow")
    st.write("✊ **Fist (2s)**: Dark Mode")
    st.write("✋ **Palm (2s)**: Light Mode")