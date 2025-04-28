import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. Tiền xử lý dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Đảm bảo các thư mục dữ liệu đúng
train_directory = 'data/train'
val_directory = 'data/val'

# Tạo generators cho dữ liệu huấn luyện và kiểm tra
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 2. Xây dựng mô hình ResNet50 với các lớp tuỳ chỉnh
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freezing base model (chỉ train phần Dense)

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile mô hình với Adam optimizer
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Các Callback hỗ trợ trong quá trình huấn luyện
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    'models/best_model3.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# 4. Huấn luyện lần 1 (Chỉ train phần Dense)
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint]
)

# 5. Fine-tune thêm (Mở khóa một phần của ResNet50)
base_model.trainable = True

# Freeze bớt các layer đầu (chỉ train từ conv5_block1 trở về sau)
fine_tune_at = 140  # Layer thứ 140 của ResNet50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile lại mô hình sau khi thay đổi trainable
model.compile(
    optimizer=optimizers.Adam(1e-5),  # Sử dụng learning rate nhỏ hơn
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Huấn luyện lần 2 (Fine-tuning)
history_fine = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint]
)

# 7. Save model cuối cùng sau khi fine-tune
model.save('models/resnet_animal3.h5')
print("Training hoàn tất! Model được lưu vào 'models/resnet_animal3.h5'")
