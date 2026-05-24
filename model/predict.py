import numpy as np
import cv2

from tensorflow.keras.models import load_model

# Load trained model
model = load_model('model/tumor_model.h5')

IMG_SIZE = 128


def predict_tumor(image_path):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image / 255.0

    image = np.reshape(image, (1, IMG_SIZE, IMG_SIZE, 3))

    prediction = model.predict(image)

    result = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    if result == 1:
        label = "Tumor Detected"
    else:
        label = "No Tumor Detected"

    return label, round(confidence, 2)