import streamlit as st
import cv2
import mediapipe as mp
import time
import av
import queue
from streamlit_webrtc import webrtc_streamer

# --- 1. UI Configuration & Branding ---
st.set_page_config(layout="centered", page_title="Gesture FX Web")
st.title("🖐️ Hand Triggered Effects")
st.caption("Developed by **Allen Charles** | [allencharles.dev](https://allencharles.dev)")

# --- 2. Thread-Safe Communication Bridge ---
# We use a global-level queue because Streamlit background threads 
# cannot access st.session_state directly.
if "gesture_queue" not in st.session_state:
    st.session_state.gesture_queue = queue.Queue()

# Reference the queue for the callback function
result_queue = st.session_state.gesture_queue

# --- 3. Setup MediaPipe ---
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# --- 4. Timer State Tracking (Internal) ---
class GestureTracker:
    def __init__(self):
        self.last_gesture = None
        self.start_time = 0
        self.done = False

# This persists inside the video thread
tracker = GestureTracker()

# --- 5. The Video Processing Engine ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    # Process Frame
    results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    this_gesture = None
    anchor_pt = None

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            pts = hand.landmark
            anchor_pt = (int(pts[0].x * w), int(pts[0].y * h))
            
            # Gesture Detection Logic (Original Logic)
            t_tip, i_tip, m_tip, r_tip, p_tip = pts[4], pts[8], pts[12], pts[16], pts[20]
            
            if t_tip.y < i_tip.y and t_tip.y < m_tip.y:
                this_gesture = "thumb"
            elif i_tip.y < pts[6].y and m_tip.y < pts[10].y:
                this_gesture = "peace"
            elif i_tip.y > pts[5].y and m_tip.y > pts[9].y:
                this_gesture = "fist"
            elif i_tip.y < pts[5].y and m_tip.y < pts[9].y:
                this_gesture = "palm"

    # --- 2-Second Timer & Progress Ring Logic ---
    if this_gesture and this_gesture == tracker.last_gesture:
        if not tracker.done:
            diff = time.time() - tracker.start_time
            
            # Draw Progress Ring on Video
            if anchor_pt:
                progress = min(1.0, diff / 2.0)
                cv2.circle(img, anchor_pt, 45, (100, 100, 100), 1)
                cv2.ellipse(img, anchor_pt, (45, 45), -90, 0, int(progress * 360), (0, 255, 120), 4)
            
            if diff >= 2.0:
                result_queue.put(this_gesture) # Send trigger to UI thread
                tracker.done = True
    else:
        tracker.last_gesture = this_gesture
        tracker.start_time = time.time()
        tracker.done = False

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 6. WebRTC Streamer Start ---
webrtc_streamer(
    key="gesture-fx-prod",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# --- 7. Effect Deployment (Main UI Thread) ---
try:
    # Check the "mailbox" for a triggered gesture
    triggered = result_queue.get_nowait()
    
    if triggered == "thumb": 
        st.balloons()
    elif triggered == "peace": 
        st.snow()
    elif triggered == "fist": 
        st.markdown("<style>.stApp { background-color: #1e1e1e; color: white; transition: 0.5s; }</style>", unsafe_allow_html=True)
    elif triggered == "palm": 
        st.markdown("<style>.stApp { background-color: white; color: black; transition: 0.5s; }</style>", unsafe_allow_html=True)
except queue.Empty:
    pass

# --- 8. Sidebar Spellbook ---
with st.sidebar:
    st.header("📖 Spellbook")
    st.write("👍 **Thumb (2s)**: Balloons")
    st.write("✌️ **Peace (2s)**: Snow")
    st.write("✊ **Fist (2s)**: Dark Mode")
    st.write("✋ **Palm (2s)**: Light Mode")
    st.divider()
    st.info("Check out my portfolio at [allencharles.dev](https://allencharles.dev)")