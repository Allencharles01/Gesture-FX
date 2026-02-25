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

# ---------- Background Layer ----------
st.markdown("""
<style>
#particles {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 9999;
  pointer-events: none;
  overflow: hidden;
}
</style>

<div id="particles"></div>
""", unsafe_allow_html=True)


# ---------- EFFECT FUNCTIONS ----------

def trigger_snow():
    st.markdown("""
    <script>
    clearParticles();

    const container = document.getElementById("particles");
    const w = window.innerWidth;
    const h = window.innerHeight;

    const count = 120;

    for (let i = 0; i < count; i++) {
        const flake = document.createElement("div");

        flake.innerText = "❄️";
        flake.style.position = "absolute";
        flake.style.fontSize = (Math.random() * 25 + 20) + "px";
        flake.style.opacity = Math.random() * 0.8 + 0.2;
        flake.style.pointerEvents = "none";

        const depth = Math.random();
        if (depth < 0.3) {
            flake.style.filter = "blur(2px)";
            flake.style.opacity = 0.4;
        }

        container.appendChild(flake);

        const xStart = Math.random() * w;
        const drift = (Math.random() - 0.5) * 200;
        const rotation = Math.random() * 360;
        const duration = 6000 + Math.random() * 6000;

        flake.animate([
          {
            transform: `translate3d(${xStart}px, -50px, 0) rotate(0deg)`,
            opacity: 0
          },
          {
            opacity: flake.style.opacity,
            offset: 0.1
          },
          {
            transform: `translate3d(${xStart + drift}px, ${h + 100}px, 0) rotate(${rotation}deg)`,
            opacity: 0
          }
        ], {
          duration: duration,
          delay: -Math.random() * duration,
          iterations: Infinity,
          easing: "linear"
        });
    }

    function clearParticles() {
        const container = document.getElementById("particles");
        container.innerHTML = "";
    }
    </script>
    """, unsafe_allow_html=True)


def trigger_confetti():
    st.markdown("""
    <script>
    const container = document.getElementById("particles");
    container.innerHTML = "";

    const w = window.innerWidth;
    const h = window.innerHeight;

    const colors = ["#ff3b3b","#3bff57","#3b8bff","#ffd93b","#ff6ec7"];
    const count = 250;

    for (let i = 0; i < count; i++) {
        const piece = document.createElement("div");

        piece.style.position = "absolute";
        piece.style.width = (Math.random()*8+4) + "px";
        piece.style.height = (Math.random()*12+6) + "px";
        piece.style.background = colors[Math.floor(Math.random()*colors.length)];
        piece.style.opacity = 0.9;
        piece.style.transform = "rotate(" + (Math.random()*360) + "deg)";
        piece.style.pointerEvents = "none";

        container.appendChild(piece);

        const xStart = Math.random() * w;
        const drift = (Math.random() - 0.5) * 300;
        const rotation = Math.random() * 720;
        const duration = 3000 + Math.random()*3000;

        piece.animate([
            { transform: `translate3d(${xStart}px, -20px, 0) rotate(0deg)` },
            { transform: `translate3d(${xStart + drift}px, ${h+30}px, 0) rotate(${rotation}deg)` }
        ], {
            duration: duration,
            delay: -Math.random()*duration,
            iterations: Infinity,
            easing: "cubic-bezier(.37,0,.63,1)"
        });
    }
    </script>
    """, unsafe_allow_html=True)


def clear_effects():
    st.markdown("""
    <script>
    const container = document.getElementById("particles");
    container.innerHTML = "";
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

# ---------- Trigger Handling ----------
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