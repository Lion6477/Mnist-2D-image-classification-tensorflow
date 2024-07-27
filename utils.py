from tensorflow.keras.datasets import mnist

def load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Нормализация изображений из диапазона [0, 255] в диапазон [0, 1]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Преобразование изображений в векторы
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    return (x_train, y_train), (x_test, y_test)
