import streamlit as st
import cv2
import mediapipe as mp
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(layout="centered", page_title="Gesture FX Web")
st.title("🖐️ Hand Triggered Effects")
st.caption("Developed by Allen Charles | allencharles.dev")

HOLD_DELAY = 2.0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = mp_hands.Hands(
            min_detection_confidence=0.7,
            max_num_hands=1,
            model_complexity=0,
        )
        self.last = None
        self.start = 0
        self.done = False
        self.trigger = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        h, w, _ = img.shape
        results = self.detector.process(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )

        this_gesture = None
        anchor_pt = None

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
                pts = hand.landmark
                anchor_pt = (int(pts[0].x * w), int(pts[0].y * h))

                t, i, m, r, p = pts[4], pts[8], pts[12], pts[16], pts[20]

                if t.y < i.y and t.y < m.y:
                    this_gesture = "thumb"
                elif i.y < pts[6].y and m.y < pts[10].y:
                    this_gesture = "peace"
                elif i.y > pts[5].y and m.y > pts[9].y:
                    this_gesture = "fist"
                elif i.y < pts[5].y and m.y < pts[9].y:
                    this_gesture = "palm"

        # -------- Timer Logic -------- #
        if this_gesture and this_gesture == self.last:
            if not self.done:
                diff = time.time() - self.start

                if anchor_pt:
                    progress = min(1.0, diff / HOLD_DELAY)
                    cv2.circle(img, anchor_pt, 40, (100, 100, 100), 1)
                    cv2.ellipse(
                        img,
                        anchor_pt,
                        (40, 40),
                        -90,
                        0,
                        int(progress * 360),
                        (0, 255, 120),
                        3,
                    )

                if diff >= HOLD_DELAY:
                    self.trigger = this_gesture
                    self.done = True
        else:
            self.last = this_gesture
            self.start = time.time()
            self.done = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")


ctx = webrtc_streamer(
    key="gesturefx",
    video_processor_factory=GestureProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ---------- Trigger Effects in Main Thread ---------- #
if ctx.video_processor:
    processor = ctx.video_processor

    if processor.trigger:
        if processor.trigger == "thumb":
            st.balloons()

        elif processor.trigger == "peace":
            st.snow()

        elif processor.trigger == "fist":
            st.markdown(
                "<style>.stApp { background-color:#1e1e1e; color:white; }</style>",
                unsafe_allow_html=True,
            )

        elif processor.trigger == "palm":
            st.markdown(
                "<style>.stApp { background-color:white; color:black; }</style>",
                unsafe_allow_html=True,
            )

        processor.trigger = None
        st.rerun()


with st.sidebar:
    st.header("📖 Spellbook")
    st.write("👍 Hold 2s → Balloons")
    st.write("✌️ Hold 2s → Snow")
    st.write("✊ Hold 2s → Dark Mode")
    st.write("✋ Hold 2s → Light Mode")