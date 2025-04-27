import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model đã train
model = load_model('models/resnet_animal.h5')

# 1. Tiền xử lý dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

# 2. Tạo train_generator để lấy các lớp đã học
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Danh sách các lớp (class_names) từ train_generator
class_names = list(train_generator.class_indices.keys())

# Hàm dự đoán động vật
def predict_animal(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = np.argmax(pred)
    return class_names[class_id]

# Test với ảnh con mèo
image_path2 = "testmeo.jpg"

# In kết quả dự đoán
print("Predicted animal:", predict_animal(image_path2))

# In ra các lớp đã học
print("Classes the model can recognize:")
for class_name, class_index in train_generator.class_indices.items():

    print(f"Class name : {class_name}, Class index: {class_index}")
