import streamlit as st
import cv2
import mediapipe as mp
import time

# Quick UI config
st.set_page_config(layout="centered", page_title="Gesture FX")
st.title("🖐️ Hand Triggered Effects")
st.caption("Hold a gesture for 2 seconds to trigger. Try: 👍, ✌️, ✊ (Dark), ✋ (Light)")

# Toggle for the loop
run_app = st.checkbox('Toggle Webcam')
video_placeholder = st.image([])

# Setup MediaPipe 
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
draw_utils = mp_draw = mp.solutions.drawing_utils

# Some laptops use 0, mine uses 1. Change if it doesn't open.
cap = cv2.VideoCapture(0) 

# State management - using session_state so it doesn't reset every rerun
if 'gesture_state' not in st.session_state:
    st.session_state.gesture_state = {"last": None, "start": 0, "done": False}

def apply_theme(dark_mode=False):
    bg = "#1e1e1e" if dark_mode else "white"
    fg = "white" if dark_mode else "black"
    st.markdown(f"""<style>.stApp {{ background-color: {bg}; color: {fg}; }}</style>""", unsafe_allow_html=True)

HOLD_DELAY = 2.0 

while run_app:
    ret, img = cap.read()
    if not ret:
        st.error("Can't find webcam. Check index?")
        break

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
            
            # Wrist point for the loading UI
            anchor_pt = (int(pts[0].x * w), int(pts[0].y * h))

            # --- Simple Gesture Detection ---
            # Tips: 4=Thumb, 8=Index, 12=Middle, 16=Ring, 20=Pinky
            t_tip, i_tip, m_tip, r_tip, p_tip = pts[4], pts[8], pts[12], pts[16], pts[20]
            
            # Basic logic for thumb up
            if t_tip.y < i_tip.y and t_tip.y < m_tip.y:
                this_gesture = "thumb"
            # Peace sign
            elif i_tip.y < pts[6].y and m_tip.y < pts[10].y and r_tip.y > pts[14].y:
                this_gesture = "peace"
            # Fist (all tips below knuckles)
            elif i_tip.y > pts[5].y and m_tip.y > pts[9].y and r_tip.y > pts[13].y and p_tip.y > pts[17].y:
                this_gesture = "fist"
            # Palm (all tips above knuckles)
            elif i_tip.y < pts[5].y and m_tip.y < pts[9].y and r_tip.y < pts[13].y and p_tip.y < pts[17].y:
                this_gesture = "palm"

    # --- Timer Logic ---
    state = st.session_state.gesture_state
    
    if this_gesture and this_gesture == state["last"]:
        if not state["done"]:
            diff = time.time() - state["start"]
            
            # Draw the progress ring
            if anchor_pt:
                # Calculate percent for the circle
                progress = min(1.0, diff / HOLD_DELAY)
                # Drawing a simple 'loading' ring
                cv2.circle(img, anchor_pt, 45, (100, 100, 100), 1) # Track
                cv2.ellipse(img, anchor_pt, (45, 45), -90, 0, int(progress * 360), (0, 255, 120), 4)
            
            if diff >= HOLD_DELAY:
                # Trigger specific Streamlit effects
                if this_gesture == "thumb": st.balloons()
                elif this_gesture == "peace": st.snow()
                elif this_gesture == "fist": apply_theme(dark_mode=True)
                elif this_gesture == "palm": apply_theme(dark_mode=False)
                
                state["done"] = True
    else:
        # Reset tracker if hand moves/changes
        st.session_state.gesture_state = {
            "last": this_gesture, 
            "start": time.time(), 
            "done": False
        }

    # Show the BGR frame back in Streamlit
    video_placeholder.image(img, channels="BGR")

cap.release()