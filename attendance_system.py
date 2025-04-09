import cv2
import dlib
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from datetime import datetime
import time
import os

# Load models
eye_model = load_model("models/eye_state_model.h5")
emotion_model = load_model("models/emotion_model.h5")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Labels
eye_labels = ["Closed", "Open"]
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Attendance variables
active_time = 0
drowsy_time = 0
start_time = time.time()
last_state = "non-drowsy"

# Helper functions
def extract_eye_regions(gray, shape):
    left_eye = gray[shape[37][1]:shape[41][1], shape[36][0]:shape[39][0]]
    right_eye = gray[shape[43][1]:shape[47][1], shape[42][0]:shape[45][0]]
    return left_eye, right_eye

def preprocess_eye(eye_img):
    if eye_img.size == 0:
        return None
    eye_img = cv2.resize(eye_img, (24, 24))
    eye_img = eye_img.astype("float") / 255.0
    eye_img = img_to_array(eye_img)
    eye_img = np.expand_dims(eye_img, axis=0)
    return eye_img

def detect_emotion(face_img):
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (48, 48))
    face_resized = face_resized.astype("float") / 255.0
    face_resized = img_to_array(face_resized)
    face_resized = np.expand_dims(face_resized, axis=0)
    emotion_prediction = emotion_model.predict(face_resized)[0]
    return emotion_labels[np.argmax(emotion_prediction)]

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        if frame is None:
            print("Empty frame received.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray is None or gray.dtype != np.uint8:
            print("Detector error: Image not in expected grayscale format.")
            continue

        try:
            faces = detector(gray, 0)
        except Exception as e:
            print("Detector error:", str(e))
            continue

        face_present = len(faces) > 0
        current_time = time.time()

        state = last_state

        for face in faces:
            shape = predictor(gray, face)
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            left_eye, right_eye = extract_eye_regions(gray, shape)
            left_eye_input = preprocess_eye(left_eye)
            right_eye_input = preprocess_eye(right_eye)

            if left_eye_input is not None and right_eye_input is not None:
                left_pred = eye_model.predict(left_eye_input)[0]
                right_pred = eye_model.predict(right_eye_input)[0]
                left_label = eye_labels[np.argmax(left_pred)]
                right_label = eye_labels[np.argmax(right_pred)]

                if left_label == "Closed" and right_label == "Closed":
                    state = "drowsy"
                else:
                    state = "non-drowsy"

            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            face_img = frame[y:y+h, x:x+w]
            emotion = detect_emotion(face_img)
            cv2.putText(frame, f"{emotion}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if face_present:
            if state == "non-drowsy":
                active_time += current_time - start_time
            else:
                drowsy_time += current_time - start_time
        start_time = current_time
        last_state = state

        # Display
        cv2.putText(frame, f"State: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if state == "non-drowsy" else (0, 0, 255), 2)
        cv2.putText(frame, f"Active Time: {int(active_time)}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Drowsy Time: {int(drowsy_time)}s", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
        cv2.imshow("Drowsiness and Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"Final Active Time: {int(active_time)}s")
    print(f"Final Drowsy Time: {int(drowsy_time)}s")

