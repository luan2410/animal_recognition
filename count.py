# Xem các lớp mà mô hình đã nhận diện
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# In ra các lớp đã học
print("Classes the model can recognize:")
for class_name, class_index in train_generator.class_indices.items():
    print(f"Class name: {class_name}, Class index: {class_index}")
