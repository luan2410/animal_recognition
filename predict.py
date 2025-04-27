import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model đã train
model = load_model('models/resnet_animal.h5')

# Danh sách lớp động vật (lấy từ train_generator.class_indices)
class_names = ['cat', 'dog', 'horse', 'lion', ...]  # Thay bằng tên lớp của bạn

def predict_animal(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = np.argmax(pred)
    return class_names[class_id]

# Test
image_path = "test_dog.jpg"  # Thay bằng đường dẫn ảnh của bạn
print("Predicted animal:", predict_animal(image_path))