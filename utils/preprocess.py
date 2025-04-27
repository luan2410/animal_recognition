import cv2
import numpy as np

def load_and_preprocess(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Chuẩn hóa [0, 1]
    return img