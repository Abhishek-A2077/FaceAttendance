import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Dataset path and categories
data_path = os.path.join('data', 'mrleyedataset')
categories = ['Close-Eyes', 'Open-Eyes']
img_size = 24

# Prepare data lists
X = []
y = []

# Load and process images
for label, category in enumerate(categories):
    folder_path = os.path.join(data_path, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(label)
        except Exception as e:
            print(f"Skipped image {img_path}: {e}")

# Convert to NumPy arrays
X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
y = to_categorical(y, num_classes=2)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save model
model.save('eye_state_classifier.h5')
print("✅ Model trained and saved as 'eye_state_classifier.h5'")
