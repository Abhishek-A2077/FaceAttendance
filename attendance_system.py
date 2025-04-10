import cv2
import dlib
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from datetime import datetime
import time
import os
from flask import Flask, Response
import threading

# Initialize Flask app
app = Flask(__name__)
current_frame = None

# Load models
eye_model = load_model("eye_state_classifier.h5")
emotion_model = load_model("emotion_classifier.h5")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Labels
eye_labels = ["Closed", "Open"]
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Attendance variables
active_time = 0
drowsy_time = 0
start_time = time.time()
last_state = "non-drowsy"

# Flask routes
@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Attendance System</title>
        <style>
          body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f0f0; margin: 0; padding: 20px; }
          h1 { color: #333; }
          img { border: 3px solid #333; border-radius: 10px; max-width: 90%; }
        </style>
      </head>
      <body>
        <h1>Drowsiness and Emotion Detection</h1>
        <img src="/video_feed" width="640" height="480" />
        <p>Press Ctrl+C in the terminal to stop the application</p>
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    def generate():
        global current_frame
        while True:
            if current_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', current_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

# Start Flask server in a thread
flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False))
flask_thread.daemon = True
flask_thread.start()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Main function to handle the processing loop
def main_loop():
    global current_frame, active_time, drowsy_time, start_time, last_state
    
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
            
            # Update global frame variable for web display
            current_frame = frame.copy()

            # Check for keyboard interrupt
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cap.release()
        print(f"Final Active Time: {int(active_time)}s")
        print(f"Final Drowsy Time: {int(drowsy_time)}s")

# Run the main loop
if __name__ == "__main__":
    main_loop()
