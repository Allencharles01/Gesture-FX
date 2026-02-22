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
# We use session_state to ensure the queue and timer persist correctly
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

# --- 3. Setup MediaPipe ---
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Use session state for the tracker so it doesn't reset on every rerun
if "tracker" not in st.session_state:
    st.session_state.tracker = {"last": None, "start": 0, "done": False}

# --- 4. Video Engine (Timer & Drawing Logic) ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    this_gesture = None
    anchor_pt = None
    tracker = st.session_state.tracker

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            pts = hand.landmark
            anchor_pt = (int(pts[0].x * w), int(pts[0].y * h))
            
            # Gesture Detection
            t_tip, i_tip, m_tip, r_tip, p_tip = pts[4], pts[8], pts[12], pts[16], pts[20]
            if t_tip.y < i_tip.y and t_tip.y < m_tip.y: this_gesture = "thumb"
            elif i_tip.y < pts[6].y and m_tip.y < pts[10].y: this_gesture = "peace"
            elif i_tip.y > pts[5].y and m_tip.y > pts[9].y: this_gesture = "fist"
            elif i_tip.y < pts[5].y and m_tip.y < pts[9].y: this_gesture = "palm"

    # Timer Logic + Drawing Progress Ring
    if this_gesture and this_gesture == tracker["last"]:
        if not tracker["done"]:
            diff = time.time() - tracker["start"]
            if anchor_pt:
                progress = min(1.0, diff / 2.0)
                cv2.circle(img, anchor_pt, 45, (100, 100, 100), 1)
                cv2.ellipse(img, anchor_pt, (45, 45), -90, 0, int(progress * 360), (0, 255, 120), 4)
            
            if diff >= 2.0:
                st.session_state.result_queue.put(this_gesture)
                tracker["done"] = True
    else:
        tracker["last"] = this_gesture
        tracker["start"] = time.time()
        tracker["done"] = False

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. Start Camera (Contained) ---
# Using a single, unique key to prevent DuplicateElementKey errors
ctx = webrtc_streamer(
    key="gesture-fx-final-v1", 
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- 6. The UI Trigger (Main Thread) ---
# We check the queue once. If an effect is found, we fire it.
try:
    triggered = st.session_state.result_queue.get_nowait()
    if triggered == "thumb": st.balloons()
    elif triggered == "peace": st.snow()
    elif triggered == "fist": 
        st.markdown("<style>.stApp { background-color: #1e1e1e; color: white; transition: 0.5s; }</style>", unsafe_allow_html=True)
    elif triggered == "palm": 
        st.markdown("<style>.stApp { background-color: white; color: black; transition: 0.5s; }</style>", unsafe_allow_html=True)
except queue.Empty:
    pass

# Small button to manually refresh if the user feels stuck, 
# preventing the "infinite loading" loop.
if ctx.state.playing:
    if st.button("Refresh Effects"):
        st.rerun()

# --- 7. Sidebar Spellbook ---
with st.sidebar:
    st.header("📖 Spellbook")
    st.write("👍 **Thumb (2s)**: Balloons")
    st.write("✌️ **Peace (2s)**: Snow")
    st.write("✊ **Fist (2s)**: Dark Mode")
    st.write("✋ **Palm (2s)**: Light Mode")