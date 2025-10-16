import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from PIL import Image

# Tham số
IMG_HEIGHT, IMG_WIDTH = 100, 100
BATCH_SIZE = 16
EPOCHS = 10

# Đường dẫn dữ liệu (tạo hai thư mục train và validation, mỗi thư mục có 3 thư mục con: apple, orange, banana)
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# Tạo generator load ảnh
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True)

val_data_gen = validation_image_generator.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Xây dựng mô hình CNN đơn giản
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // BATCH_SIZE
)

# Lưu mô hình
model.save('fruit_classifier_model.h5')

# Hàm dự đoán ảnh mới
class_names = ['apple', 'banana', 'orange']

def predict_image(image_path):
    img = Image.open(image_path).resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)  # thêm batch dimension
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    print(f'Predicted: {predicted_class} with confidence {confidence:.2f}')

# Ví dụ gọi dự đoán
# predict_image('dataset/validation/apple/apple1.jpg')
