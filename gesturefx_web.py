import streamlit as st
import cv2
import mediapipe as mp
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- UI Configuration ---
st.set_page_config(layout="centered", page_title="Gesture FX Web")
st.title("🖐️ Gesture FX: Web Edition")
st.markdown("""
    This version is optimized for the web. **Allow camera access** when prompted.
    **Goal:** Hold a gesture for 2 seconds to trigger an effect!
""")

# Setup MediaPipe 
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# CSS for Dynamic Theme Switching
def apply_theme(dark_mode=False):
    bg = "#1e1e1e" if dark_mode else "white"
    fg = "white" if dark_mode else "black"
    st.markdown(f"""
        <style>
        .stApp {{ background-color: {bg}; color: {fg}; transition: background-color 0.5s; }}
        </style>
        """, unsafe_allow_html=True)

# --- Video Processing Class ---
class GestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.state = {"last": None, "start": 0, "done": False, "trigger": None}
        self.hold_delay = 2.0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb_img)

        this_gesture = None
        anchor_pt = None

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
                pts = hand.landmark
                
                # Wrist point for the progress UI
                anchor_pt = (int(pts[0].x * w), int(pts[0].y * h))

                # Gesture Logic
                t_tip, i_tip, m_tip, r_tip, p_tip = pts[4], pts[8], pts[12], pts[16], pts[20]
                
                if t_tip.y < i_tip.y and t_tip.y < m_tip.y:
                    this_gesture = "thumb"
                elif i_tip.y < pts[6].y and m_tip.y < pts[10].y and r_tip.y > pts[14].y:
                    this_gesture = "peace"
                elif i_tip.y > pts[5].y and m_tip.y > pts[9].y and r_tip.y > pts[13].y and p_tip.y > pts[17].y:
                    this_gesture = "fist"
                elif i_tip.y < pts[5].y and m_tip.y < pts[9].y and r_tip.y < pts[13].y and p_tip.y < pts[17].y:
                    this_gesture = "palm"

        # Timer logic to ensure the gesture is intentional
        if this_gesture and this_gesture == self.state["last"]:
            if not self.state["done"]:
                diff = time.time() - self.state["start"]
                
                # Visual Feedback: Progress Ring
                if anchor_pt:
                    progress = min(1.0, diff / self.hold_delay)
                    cv2.circle(img, anchor_pt, 45, (180, 180, 180), 2)
                    cv2.ellipse(img, anchor_pt, (45, 45), -90, 0, int(progress * 360), (0, 255, 120), 5)
                
                if diff >= self.hold_delay:
                    self.state["trigger"] = this_gesture
                    self.state["done"] = True
        else:
            # Reset state if hand is lost or gesture changes
            self.state = {"last": this_gesture, "start": time.time(), "done": False, "trigger": None}

        return img

# --- Main App Execution ---
# Start the WebRTC streamer
ctx = webrtc_streamer(
    key="gesture-fx-web", 
    video_transformer_factory=GestureTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # Essential for web connectivity
    media_stream_constraints={"video": True, "audio": False}
)

# Effect Triggering (Happens in the main Streamlit thread)
if ctx.video_transformer:
    triggered = ctx.video_transformer.state.get("trigger")
    if triggered:
        if triggered == "thumb": 
            st.balloons()
        elif triggered == "peace": 
            st.snow()
        elif triggered == "fist": 
            apply_theme(dark_mode=True)
        elif triggered == "palm": 
            apply_theme(dark_mode=False)
        
        # Reset the trigger so it doesn't fire repeatedly
        ctx.video_transformer.state["trigger"] = None

# Sidebar Instructions
with st.sidebar:
    st.header("Spellbook")
    st.write("👍 **Thumb Up**: Balloons")
    st.write("✌️ **Peace Sign**: Snow")
    st.write("✊ **Fist**: Dark Mode")
    st.write("✋ **Open Palm**: Light Mode")
    st.divider()
    st.info("Check out my portfolio at [allencharles.dev](https://allencharles.dev)")