import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("emotion_classifier_final.h5")


# Emotion labels used during training (FER2013 has 7 classes)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Path to test data
test_data_dir = "data/emotion detection/test"

# Loop through each emotion folder
for emotion_folder in os.listdir(test_data_dir):
    folder_path = os.path.join(test_data_dir, emotion_folder)
    
    if not os.path.isdir(folder_path):
        continue  # Skip if not a folder

    print(f"\n--- Testing images in folder: {emotion_folder} ---")
    
    for img_file in os.listdir(folder_path):
        if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_img, (48, 48))
        img_array = img_to_array(resized_img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize

        predictions = model.predict(img_array)
        predicted_label = emotion_labels[np.argmax(predictions)]

        print(f"{img_file} → Predicted Emotion: {predicted_label}")
