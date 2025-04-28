from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

# Load model
model = load_model('models/resnet_animal3.h5')

# Chuẩn bị danh sách class
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
class_names = list(train_generator.class_indices.keys())

# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# API dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = np.argmax(pred)
    predicted_class = class_names[class_id]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
