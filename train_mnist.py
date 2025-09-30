# train_mnist.py
import numpy as np
from tensorflow.keras.datasets import mnist
from model_mnist import create_cnn
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    model = create_cnn(input_shape=x_train.shape[1:], num_classes=10)
    model.summary()
    ckpt = ModelCheckpoint(os.path.join(MODEL_DIR, 'mnist_cnn.h5'), save_best_only=True, monitor='val_accuracy')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(x_train, y_train, validation_split=0.1, epochs=20, batch_size=128, callbacks=[ckpt, es])
    loss, acc = model.evaluate(x_test, y_test)
    print("Test acc:", acc)
    model.save(os.path.join(MODEL_DIR, "mnist_cnn_final.h5"))

if __name__ == "__main__":
    main()
