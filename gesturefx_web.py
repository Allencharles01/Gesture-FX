import streamlit as st
import cv2
import mediapipe as mp
import time
import av
import random
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
        self.active_effect = None
        self.particles = []

    def create_particles(self, w, h, count=50):
        self.particles = [
            [random.randint(0, w), random.randint(-h, 0), random.randint(2, 6)]
            for _ in range(count)
        ]

    def draw_snow(self, img):
        h, w, _ = img.shape
        for p in self.particles:
            cv2.circle(img, (p[0], p[1]), p[2], (255, 255, 255), -1)
            p[1] += 3
            if p[1] > h:
                p[1] = random.randint(-20, 0)

    def draw_confetti(self, img):
        h, w, _ = img.shape
        colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
        for p in self.particles:
            cv2.circle(img, (p[0], p[1]), p[2], random.choice(colors), -1)
            p[1] += 5
            if p[1] > h:
                p[1] = random.randint(-20, 0)

    def apply_dark_filter(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0,0), (img.shape[1], img.shape[0]), (0,0,0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

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

        # Timer logic
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
                    self.active_effect = this_gesture
                    self.done = True
                    self.create_particles(w, h)
        else:
            self.last = this_gesture
            self.start = time.time()
            self.done = False

        # Apply active effect
        if self.active_effect == "peace":
            self.draw_snow(img)
        elif self.active_effect == "thumb":
            self.draw_confetti(img)
        elif self.active_effect == "fist":
            self.apply_dark_filter(img)
        elif self.active_effect == "palm":
            self.active_effect = None

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="gesturefx",
    video_processor_factory=GestureProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)