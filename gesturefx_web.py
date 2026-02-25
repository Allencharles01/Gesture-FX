import streamlit as st
import cv2
import mediapipe as mp
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(layout="centered", page_title="Gesture FX")
st.title("🖐️ Gesture FX")
st.caption("Developed by Allen Charles | allencharles.dev")

HOLD_DELAY = 2.0

# ---------- Background Animation Layer ----------
st.markdown("""
<style>
#universe {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 9999;
  pointer-events: none;
  overflow: hidden;
}
.star {
  position: absolute;
  border-radius: 50%;
}
</style>

<div id="universe"></div>
""", unsafe_allow_html=True)


# ---------- JS Effect Triggers ----------
def trigger_snow():
    st.markdown("""
    <script>
    const universe = document.getElementById("universe");
    universe.innerHTML = "";
    const count = 200;
    const w = window.innerWidth;
    const h = window.innerHeight;

    for (let i = 0; i < count; i++) {
        const el = document.createElement("div");
        el.className = "star";
        el.style.width = el.style.height = (Math.random()*4+2) + "px";
        el.style.background = "white";
        universe.appendChild(el);

        const xStart = Math.random() * w;
        const duration = 5000 + Math.random()*3000;

        el.animate([
            { transform: `translate3d(${xStart}px, -10px, 0)`, opacity: 0 },
            { opacity: 0.8, offset: 0.1 },
            { transform: `translate3d(${xStart}px, ${h+20}px, 0)`, opacity: 0 }
        ], {
            duration: duration,
            delay: -Math.random()*duration,
            iterations: Infinity,
            easing: "linear"
        });
    }
    </script>
    """, unsafe_allow_html=True)


def trigger_confetti():
    st.markdown("""
    <script>
    const universe = document.getElementById("universe");
    universe.innerHTML = "";
    const count = 200;
    const w = window.innerWidth;
    const h = window.innerHeight;
    const colors = ["#ff3b3b","#3bff57","#3b8bff","#ffd93b"];

    for (let i = 0; i < count; i++) {
        const el = document.createElement("div");
        el.className = "star";
        el.style.width = el.style.height = (Math.random()*6+4) + "px";
        el.style.background = colors[Math.floor(Math.random()*colors.length)];
        universe.appendChild(el);

        const xStart = Math.random() * w;
        const duration = 3000 + Math.random()*2000;

        el.animate([
            { transform: `translate3d(${xStart}px, -10px, 0)` },
            { transform: `translate3d(${xStart}px, ${h+20}px, 0)` }
        ], {
            duration: duration,
            delay: -Math.random()*duration,
            iterations: Infinity,
            easing: "linear"
        });
    }
    </script>
    """, unsafe_allow_html=True)


def clear_effects():
    st.markdown("""
    <script>
    const universe = document.getElementById("universe");
    universe.innerHTML = "";
    </script>
    """, unsafe_allow_html=True)


def apply_dark():
    st.markdown("""
    <style>
    body { background-color: #111 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)


def apply_light():
    st.markdown("""
    <style>
    body { background-color: white !important; color: black !important; }
    </style>
    """, unsafe_allow_html=True)


# ---------- MediaPipe Setup ----------
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

# ---------- Main Thread Effect Handling ----------
if ctx.video_processor:
    processor = ctx.video_processor
    if processor.trigger:

        if processor.trigger == "thumb":
            trigger_confetti()

        elif processor.trigger == "peace":
            trigger_snow()

        elif processor.trigger == "fist":
            apply_dark()

        elif processor.trigger == "palm":
            apply_light()
            clear_effects()

        processor.trigger = None