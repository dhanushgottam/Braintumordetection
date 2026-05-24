import os
import numpy as np
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Dataset paths
yes_path = '../dataset/yes'
no_path = '../dataset/no'

# Image size
IMG_SIZE = 128

X = []
y = []

# Load tumor images
for image_name in os.listdir(yes_path):

    image_path = os.path.join(yes_path, image_name)

    image = cv2.imread(image_path)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    X.append(image)

    y.append(1)

# Load non-tumor images
for image_name in os.listdir(no_path):

    image_path = os.path.join(no_path, image_name)

    image = cv2.imread(image_path)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    X.append(image)

    y.append(0)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize images
X = X / 255.0

# Convert labels
y = to_categorical(y, num_classes=2)

# Create CNN model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(
    X,
    y,
    epochs=5,
    batch_size=16,
    validation_split=0.2
)

# Save model
model.save('tumor_model.h5')

print("Model Trained and Saved Successfully")