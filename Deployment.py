# streamlit_gesture_tuner_holdmode.py
import streamlit as st
import cv2
import time
import collections
import numpy as np
import pyautogui
from ultralytics import YOLO

# ---------- config / defaults ----------
MODEL_PATH = r"runs\detect\\train2\weights\best.pt"
CAM_INDEX = 0
DEFAULTS = {
    "FPS_PROCESS": 15,
    "CONF_THRESH": 0.50,
    "CONFIRM_COUNT": 3,
    "COOLDOWN_SECS": 0.35,
    "KEY_PRESS_DURATION": 0.06
}
CLASS_TO_KEY = {
    "High Five": "up",
    "Two": "down",
    "Thumb": "left",
    "Pinky": "right",
    "Flat": "space"
}
pyautogui.FAILSAFE = False

# ---------- load model once ----------
@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ---------- UI ----------
st.title("Hand Gesture Controller — Tuner (Hold mode supported)")
st.markdown("Toggle **Hold mode** if the target game requires you to hold a key to keep moving. When Hold mode is on, keys are held down while the gesture is present.")

col1, col2 = st.columns(2)

with col1:
    fps_process = st.slider("FPS (processing)", 5, 30, DEFAULTS["FPS_PROCESS"], step=1,
                            help="How many frames per second to process (lower -> less CPU, more lag).")
    conf_thresh = st.slider("Confidence threshold", 0.1, 0.99, float(DEFAULTS["CONF_THRESH"]), step=0.01,
                            help="Ignore detections below this confidence.")
    confirm_count = st.slider("Confirm count (frames)", 1, 7, DEFAULTS["CONFIRM_COUNT"],
                              help="How many recent frames must agree before action fires.")
with col2:
    cooldown_secs = st.slider("Cooldown (s)", 0.0, 1.5, float(DEFAULTS["COOLDOWN_SECS"]), step=0.01,
                              help="Minimum seconds between actions to avoid spam. Ignored in Hold mode.")
    key_press_duration = st.slider("Key press duration (s)", 0.01, 0.3, float(DEFAULTS["KEY_PRESS_DURATION"]), step=0.01,
                                   help="How long to hold the key down for tap mode.")
    show_preview = st.checkbox("Show camera preview", value=True)

hold_mode = st.checkbox("Hold mode (for continuous movement)", value=False,
                        help="If on: keyDown when gesture confirmed, keyUp when gesture stops. If off: short tap behavior.")
# Activate checkbox uses a key so we can detect real-time toggle in the loop via session_state
st.checkbox("Activate Hand Gesture Controller", key="activate", value=False)

# placeholders
frame_ph = st.empty()
status_ph = st.empty()
recent_ph = st.empty()

# ---------- runtime variables ----------
# We'll recreate recent_labels inside the loop whenever confirm_count changes; so start with something reasonable
recent_labels = collections.deque(maxlen=max(1, confirm_count))
last_action_time = 0.0
current_held_key = None  # For hold mode: which key is currently held down

def tap_key(key, duration):
    try:
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)
    except Exception as e:
        st.write("pyautogui error (tap):", e)

def hold_key_down(key):
    try:
        pyautogui.keyDown(key)
    except Exception as e:
        st.write("pyautogui error (keydown):", e)

def release_key(key):
    try:
        pyautogui.keyUp(key)
    except Exception as e:
        st.write("pyautogui error (keyup):", e)

# ---------- main loop ----------
# Use activate from session_state to allow real-time toggling
if st.session_state.get("activate", False):
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        st.error("Cannot open webcam. Check CAM_INDEX or permissions.")
    else:
        st.info("Model running. Make gestures in view of the camera. Make sure the game window has focus to receive keys.")
        try:
            prev_time = 0.0
            # main loop
            while st.session_state.get("activate", False):
                # Update recent_labels if confirm_count changed
                if recent_labels.maxlen != max(1, confirm_count):
                    recent_labels = collections.deque(maxlen=max(1, confirm_count))

                # throttle to fps_process
                now_time = time.time()
                if now_time - prev_time < 1.0 / max(1, fps_process):
                    time.sleep(max(0.001, 1.0 / fps_process - (now_time - prev_time)))
                prev_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    st.warning("Frame capture failed.")
                    break

                frame = cv2.flip(frame, 1)  # mirror

                # inference
                results = model(frame, imgsz=160)
                r = results[0]

                chosen_label = None
                if r.boxes is not None and len(r.boxes) > 0:
                    confs = r.boxes.conf.cpu().numpy().flatten()
                    cls_ids = r.boxes.cls.cpu().numpy().flatten().astype(int)
                    best_idx = int(np.argmax(confs))
                    best_conf = float(confs[best_idx])
                    best_cls = int(cls_ids[best_idx])
                    if best_conf >= conf_thresh:
                        cls_name = model.names.get(best_cls, str(best_cls))
                        chosen_label = cls_name
                        # draw rectangle
                        xyxy = r.boxes.xyxy.cpu().numpy()[best_idx]
                        x1, y1, x2, y2 = xyxy.astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(frame, f"{cls_name} {best_conf:.2f}", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                recent_labels.append(chosen_label if chosen_label else "none")
                most_common = collections.Counter(recent_labels).most_common(1)[0][0]

                tnow = time.time()
                if hold_mode:
                    # --- HOLD MODE logic ---
                    # If a valid gesture is present and confirmed:
                    if most_common != "none" and recent_labels.count(most_common) >= max(1, confirm_count):
                        key = CLASS_TO_KEY.get(most_common, None)
                        # If it's a different key than currently held, switch
                        if key and key != current_held_key:
                            # release previous if any
                            if current_held_key is not None:
                                release_key(current_held_key)
                                status_ph.info(f"Released '{current_held_key}'")
                            # hold new
                            hold_key_down(key)
                            current_held_key = key
                            status_ph.success(f"[{time.strftime('%H:%M:%S')}] Holding '{key}' for gesture '{most_common}'")
                        # else: same key already held -> do nothing
                    else:
                        # no confirmed gesture -> release held key if exists
                        if current_held_key is not None:
                            release_key(current_held_key)
                            status_ph.info(f"Released '{current_held_key}' (no gesture)")
                            current_held_key = None
                else:
                    # --- TAP MODE logic (original behavior) ---
                    if most_common != "none" and recent_labels.count(most_common) >= max(1, confirm_count):
                        if tnow - last_action_time >= cooldown_secs:
                            key = CLASS_TO_KEY.get(most_common, None)
                            if key:
                                status_ph.info(f"[{time.strftime('%H:%M:%S')}] Gesture '{most_common}' -> tap '{key}'")
                                tap_key(key, key_press_duration)
                                last_action_time = tnow
                                recent_labels.clear()
                            else:
                                status_ph.warning(f"Detected '{most_common}' but no key mapped.")

                # show preview and UI hints
                if show_preview:
                    cv2.putText(frame, f"Most: {most_common}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                    frame_ph.image(frame, channels="BGR")

                # show recent label bar (simple)
                counts = collections.Counter(recent_labels)
                recent_ph.markdown("**Recent votes:** " + " | ".join(f"{k}:{v}" for k,v in counts.items()))

            # end while (user toggled off)
        except Exception as e:
            st.error(f"Stopped due to error: {e}")
        finally:
            # ensure any held key is released
            if current_held_key is not None:
                try:
                    release_key(current_held_key)
                except Exception:
                    pass
                current_held_key = None
            cap.release()
            frame_ph.empty()
            status_ph.empty()
            recent_ph.empty()
            st.info("Stopped. Camera released and keys (if any) released.")
else:
    st.info("Controller is inactive. Check 'Activate Hand Gesture Controller' to start.")
