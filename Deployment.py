import streamlit as st
import cv2
import time
import collections
import numpy as np
import pyautogui
from ultralytics import YOLO

ModelPath = r"runs\detect\\train2\weights\best.pt"
CameraIndex = 0

DefaultConfiguration = {
    "FpsProcessing": 15,
    "ConfidenceThreshold": 0.50,
    "MinimumFrameConfirm": 3,
    "ActionCooldown": 0.35,
    "PressDuration": 0.06
}

ClassToKeyConversion = {
    "High Five": "up",
    "Two": "down",
    "Thumb": "left",
    "Pinky": "right",
    "Flat": "space"
}

pyautogui.FAILSAFE = False
Model = YOLO(ModelPath)

st.title(":rainbow[Hand Gesture Based Game Controller]")
st.write("yay")

st.header("Settings!")

with st.expander("Open Settings"):
    FpsProcessing = st.slider(
        "FPS (processing)", 
        5, 30, 
        DefaultConfiguration["FpsProcessing"], 
        step=1,
        help="Higher FPS helps the system respond faster and prevent lag, but it costs GPU/CPU power"
    )

    ConfidenceThreshold = st.slider(
        "Confidence Threshold", 
        0.01, 0.99, 
        float(DefaultConfiguration["ConfidenceThreshold"]), 
        step=0.01,
        help="Higher confidence helps lower the system's false gesture reading"
    )

    MinimumFrameConfirm = st.slider(
        "Minimum Frame Count", 
        1, 10, 
        DefaultConfiguration["MinimumFrameConfirm"],
        help="Higher Minimum Frame Count helps prevent accidental action, but also make the system respond slower"
    )

    ActionCooldown = st.slider(
        "Action Cooldown", 
        0.0, 1.5, 
        float(DefaultConfiguration["ActionCooldown"]), 
        step=0.01,
        help="Higher cooldown helps prevent input spamming or constant hold"
    )

    PressDuration = st.slider(
        "Key Hold Duration", 
        0.01, 0.3, 
        float(DefaultConfiguration["PressDuration"]), 
        step=0.01,
        help="Some games require the key to be held briefly or the input won't register"
    )
    
    PreviewMode = st.checkbox("Show camera preview", value=True)


st.header("Mode")

with st.expander("Open Modes"):

    HoldMode = st.checkbox("Hold Mode", value=False,
                            help="Activate \"Hold Mode\" for games that requires certain keys to be held")

st.header("Activate Controller!")

with st.expander("Toggle Controller"):
    st.checkbox("Activate!", key="activate", value=False)

Frame = st.empty()
Status = st.empty()
Recent = st.empty()

RecentLabels = collections.deque(maxlen=max(1, MinimumFrameConfirm))
LastActionTime = 0.0
CurrentlyHeldKey = None

def PressKey(key, duration):
    try:
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)
    except Exception as e:
        st.write("Press Error Detected!", e)

def HoldKey(key):
    try:
        pyautogui.keyDown(key)
    except Exception as e:
        st.write("Hold Error Detected!", e)

def ReleaseKey(key):
    try:
        pyautogui.keyUp(key)
    except Exception as e:
        st.write("Release Error Deteected!", e)


if st.session_state.get("activate", False):
    cap = cv2.VideoCapture(CameraIndex)

    if not cap.isOpened():
        st.error("Unable to activate camera!")
    
    else:
        st.info("Try making gestures inside the window!")
        try:
            PastTime = 0.0
            while st.session_state.get("activate", False):

                if RecentLabels.maxlen != max(1, MinimumFrameConfirm):
                    RecentLabels = collections.deque(maxlen=max(1, MinimumFrameConfirm))

                # throttle to FpsProcessing
                PresentTime = time.time()
                if PresentTime - PastTime < 1.0 / max(1, FpsProcessing):
                    time.sleep(max(0.001, 1.0 / FpsProcessing - (PresentTime - PastTime)))
                PastTime = time.time()

                ret, frame = cap.read()
                if not ret:
                    st.warning("Frame capture failed.")
                    break

                frame = cv2.flip(frame, 1) 

                Output = Model(frame, imgsz=160)
                r = Output[0]

                ChosenLabel = None
                if r.boxes is not None and len(r.boxes) > 0:
                    Confidence = r.boxes.conf.cpu().numpy().flatten()
                    ClassID = r.boxes.cls.cpu().numpy().flatten().astype(int)
                    BestIndex = int(np.argmax(Confidence))
                    BestConfidence = float(Confidence[BestIndex])
                    BestClass = int(ClassID[BestIndex])
                    if BestConfidence >= ConfidenceThreshold:
                        ClassName = Model.names.get(BestClass, str(BestClass))
                        ChosenLabel = ClassName

                        xyxy = r.boxes.xyxy.cpu().numpy()[BestIndex]
                        x1, y1, x2, y2 = xyxy.astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"{ClassName} {BestConfidence:.2f}", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                RecentLabels.append(ChosenLabel if ChosenLabel else "none")
                MostCommon = collections.Counter(RecentLabels).most_common(1)[0][0]

                TimeRightNow = time.time()

                if HoldMode:
                    if MostCommon != "none" and RecentLabels.count(MostCommon) >= max(1, MinimumFrameConfirm):
                        key = ClassToKeyConversion.get(MostCommon, None)
                        if key and key != CurrentlyHeldKey:
                            if CurrentlyHeldKey is not None:
                                ReleaseKey(CurrentlyHeldKey)
                                Status.info(f"Released '{CurrentlyHeldKey}'")
                            HoldKey(key)
                            CurrentlyHeldKey = key
                            Status.success(f"[{time.strftime('%H:%M:%S')}] Holding '{key}' for gesture '{MostCommon}'")
                    else:
                        if CurrentlyHeldKey is not None:
                            ReleaseKey(CurrentlyHeldKey)
                            Status.info(f"Released '{CurrentlyHeldKey}' (no gesture)")
                            CurrentlyHeldKey = None

                else:
                    if MostCommon != "none" and RecentLabels.count(MostCommon) >= max(1, MinimumFrameConfirm):
                        if TimeRightNow - LastActionTime >= ActionCooldown:
                            key = ClassToKeyConversion.get(MostCommon, None)
                            if key:
                                Status.info(f"[{time.strftime('%H:%M:%S')}] Gesture '{MostCommon}' -> tap '{key}'")
                                PressKey(key, PressDuration)
                                LastActionTime = TimeRightNow
                                RecentLabels.clear()
                            else:
                                Status.warning(f"Detected '{MostCommon}' but no key mapped.")

                if PreviewMode:
                    cv2.putText(frame, f"Most: {MostCommon}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                    Frame.image(frame, channels="BGR")

                counts = collections.Counter(RecentLabels)
                Recent.markdown("**Recent votes:** " + " | ".join(f"{k}:{v}" for k,v in counts.items()))

        except Exception as e:
            st.error(f"Stopped due to error: {e}")
        finally:
            if CurrentlyHeldKey is not None:
                try:
                    ReleaseKey(CurrentlyHeldKey)
                except Exception:
                    pass
                CurrentlyHeldKey = None
            cap.release()
            Frame.empty()
            Status.empty()
            Recent.empty()
            st.info("Stopped. Camera released and keys (if any) released.")
else:
    st.info("Controller is inactive. Check 'Activate!' to start.")
