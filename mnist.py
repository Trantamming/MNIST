import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import tensorflow as tf
import tensorflowjs as tfjs
import tensorflow as tf
import numpy as np

layers = tf.keras.layers
models = tf.keras.models


# Tải dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Chuẩn hóa dữ liệu
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=-1)  # Thêm 1 kênh màu (grayscale)
x_test = np.expand_dims(x_test, axis=-1)

# Xây dựng mô hình CNN đơn giản
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 lớp đầu ra cho các số từ 0-9
])

# Biên dịch mô hình
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Xuất mô hình sang TensorFlow.js
tfjs.converters.save_keras_model(model, "mnist_tfjs_model")
