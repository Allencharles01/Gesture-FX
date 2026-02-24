import streamlit as st
import cv2
import mediapipe as mp
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(layout="centered", page_title="Gesture FX Web")
st.title("🖐️ Hand Triggered Effects")
st.caption("Developed by Allen Charles")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class GestureProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = mp_hands.Hands(
            min_detection_confidence=0.7,
            max_num_hands=1,
            model_complexity=0  # IMPORTANT: lighter model
        )
        self.last = None
        self.start = 0
        self.done = False
        self.trigger = None

    def transform(self, frame):
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

        # Timer logic (INSIDE class only)
        if this_gesture and this_gesture == self.last:
            if not self.done:
                diff = time.time() - self.start

                if anchor_pt:
                    progress = min(1.0, diff / 2.0)
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

                if diff >= 2.0:
                    self.trigger = this_gesture
                    self.done = True
        else:
            self.last = this_gesture
            self.start = time.time()
            self.done = False

        return img


ctx = webrtc_streamer(
    key="gesturefx",
    video_processor_factory=GestureProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# 🔥 Main thread safely reads trigger
if ctx.video_processor:
    trigger = ctx.video_processor.trigger

    if trigger == "thumb":
        st.balloons()
        ctx.video_processor.trigger = None

    elif trigger == "peace":
        st.snow()
        ctx.video_processor.trigger = None

    elif trigger == "fist":
        st.markdown(
            "<style>.stApp { background-color: #1e1e1e; color: white; }</style>",
            unsafe_allow_html=True,
        )
        ctx.video_processor.trigger = None

    elif trigger == "palm":
        st.markdown(
            "<style>.stApp { background-color: white; color: black; }</style>",
            unsafe_allow_html=True,
        )
        ctx.video_processor.trigger = None