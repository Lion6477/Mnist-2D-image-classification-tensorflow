import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from utils import load_dataset

# Загрузка датасета
(x_train, y_train), (x_test, y_test) = load_dataset()

# Параметры обучения
epochs = 5
learning_rate = 0.001

# Создание модели
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define batch size
batch_size = 4  # Example value, adjust as needed

# Train the model with batch size
history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)

# Получение финальных значений loss и accuracy
final_loss, final_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("Training completed")
name = ("trained_model" +
        f"_epochs-{epochs}" +
        f"_rate-{learning_rate}" +
        "_loss-" + str(round(final_loss, 3)) +
        "_acc-" + str(round(final_accuracy, 3)) + ".keras")
print(f"Save model into file \"{name}\"? y/n")

if input().lower() == "n":
    print("Model not saved")

else:
    print(f"Saving as {name}")
    model.save(name)
    print("Model saved")