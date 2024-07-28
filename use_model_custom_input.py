import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image , ImageOps



def preprocess_image(image_path):
    """
    Функция для загрузки и предобработки изображения.
    """
    # Загрузка изображения
    image = Image.open(image_path).convert('L')  # Конвертация в оттенки серого
    image = image.resize((28, 28))  # Изменение размера до 28x28

    # Конвертация изображения в numpy массив
    image_array = 1 - (np.array(image).astype(np.float32) / 255.0)
    image_array = image_array.reshape(1, -1)  # Приведение к форме (1, 784)

    return image_array,image


# Загрузка модели
# model_path = input("Enter the path of the saved model: ")
# model = tf.keras.models.load_model(model_path)
model = tf.keras.models.load_model("trained_model_epochs-5_rate-0.001_loss-0.098_acc-0.976.keras")

# Предобработка пользовательского изображения
image_path = "2.png"
processed_image,image = preprocess_image(image_path)

inverted_image = ImageOps.invert(image)

# Прогон изображения через модель для предсказания
prediction = model.predict(processed_image)
predicted_class = np.argmax(prediction)

print(f"Predicted class for the image: {predicted_class}")

# Визуализация результата
plt.imshow(inverted_image , cmap="Greys")
plt.title(f"ANN suggests the number is: {predicted_class}")
plt.show()
