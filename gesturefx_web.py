import streamlit as st
import cv2
import mediapipe as mp
import time
import av
import random
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(layout="centered", page_title="Gesture FX")
st.title("🖐️ Gesture FX")
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

        self.snowflakes = []
        self.confetti = []

    def create_snow(self, width, height):
        self.snowflakes = []
        for _ in range(120):
            self.snowflakes.append({
                "x": random.randint(0, width),
                "y": random.randint(-height, 0),
                "speed": random.uniform(1, 3),
                "size": random.uniform(0.5, 1.5),
                "drift": random.uniform(-1, 1)
            })

    def create_confetti(self, width, height):
        self.confetti = []
        colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255)]
        for _ in range(150):
            self.confetti.append({
                "x": random.randint(0, width),
                "y": random.randint(-height, 0),
                "speed": random.uniform(3,6),
                "color": random.choice(colors),
                "angle": random.randint(0,360)
            })

    def draw_snow(self, img):
        h, w, _ = img.shape
        for flake in self.snowflakes:
            flake["y"] += flake["speed"]
            flake["x"] += flake["drift"]

            if flake["y"] > h:
                flake["y"] = random.randint(-50, 0)
                flake["x"] = random.randint(0, w)

            cv2.putText(
                img,
                "❄",
                (int(flake["x"]), int(flake["y"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                flake["size"],
                (255,255,255),
                2,
                cv2.LINE_AA
            )

    def draw_confetti(self, img):
        h, w, _ = img.shape
        for piece in self.confetti:
            piece["y"] += piece["speed"]

            if piece["y"] > h:
                piece["y"] = random.randint(-50, 0)
                piece["x"] = random.randint(0, w)

            cv2.circle(
                img,
                (int(piece["x"]), int(piece["y"])),
                4,
                piece["color"],
                -1
            )

    def apply_dark(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0,0), (img.shape[1], img.shape[0]), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        h, w, _ = img.shape
        results = self.detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        this_gesture = None
        anchor_pt = None

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
                pts = hand.landmark
                anchor_pt = (int(pts[0].x * w), int(pts[0].y * h))

                t, i, m = pts[4], pts[8], pts[12]

                if t.y < i.y and t.y < m.y:
                    this_gesture = "thumb"
                elif i.y < pts[6].y and m.y < pts[10].y:
                    this_gesture = "peace"
                elif i.y > pts[5].y and m.y > pts[9].y:
                    this_gesture = "fist"
                elif i.y < pts[5].y and m.y < pts[9].y:
                    this_gesture = "palm"

        if this_gesture and this_gesture == self.last:
            if not self.done:
                diff = time.time() - self.start

                if anchor_pt:
                    progress = min(1.0, diff / HOLD_DELAY)
                    cv2.circle(img, anchor_pt, 40, (100,100,100), 1)
                    cv2.ellipse(
                        img,
                        anchor_pt,
                        (40,40),
                        -90,
                        0,
                        int(progress*360),
                        (0,255,120),
                        3
                    )

                if diff >= HOLD_DELAY:
                    self.active_effect = this_gesture
                    self.done = True

                    if this_gesture == "peace":
                        self.create_snow(w, h)
                    elif this_gesture == "thumb":
                        self.create_confetti(w, h)

        else:
            self.last = this_gesture
            self.start = time.time()
            self.done = False

        if self.active_effect == "peace":
            self.draw_snow(img)
        elif self.active_effect == "thumb":
            self.draw_confetti(img)
        elif self.active_effect == "fist":
            self.apply_dark(img)
        elif self.active_effect == "palm":
            self.active_effect = None

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="gesturefx",
    video_processor_factory=GestureProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)