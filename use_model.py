import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

import utils

# Загрузка натренированной модели
model = utils.load_model("trained_model_epochs-20_rate-0.0309_loss-0.09_acc-0.112.keras")

# Загрузка изображения
test_image = 255 - cv2.imread("5.png", cv2.IMREAD_GRAYSCALE)  # Загружаем в оттенках серого
test_image = test_image.reshape(1, 784) / 255.0  # Преобразование изображения в вектор

# Прогнозирование
output = model.predict(test_image)

# Визуализация результата
plt.imshow(test_image.reshape(28, 28), cmap="Greys")
plt.title(f"ANN suggests the number is: {output.argmax()}")
plt.show()
