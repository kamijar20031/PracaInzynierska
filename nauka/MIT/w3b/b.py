import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

keras.utils.set_random_seed(42)

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
labels = ["T-shirt", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

x_train = x_train/255.0
x_test = x_test/255.0

input = keras.Input(shape=(28,28))
h = keras.layers.Flatten()(input)
h= keras.layers.Dense(256, activation="relu", name="Hidden")(h)
output = keras.layers.Dense(10, activation="softmax", name="Output")(h)
model=keras.Model(input, output)

# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)        [(None, 28, 28)]          0         
#                                                                  
#  flatten (Flatten)           (None, 784)               0         
#                                                                  
#  Hidden (Dense)              (None, 256)               200960    
#                                                                  
#  Output (Dense)              (None, 10)                2570      
#                                                                  
# =================================================================
# Total params: 203530 (795.04 KB)
# Trainable params: 203530 (795.04 KB)
# Non-trainable params: 0 (0.00 Byte)

# Ilosc parametrow - (28*28*256 + 256) + (256*10+10) = 203530

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.2)

history_dict = history.history
loss = history_dict["loss"]
val_loss = history_dict["val_loss"]
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
epochs = range(1, len(loss)+1)
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("acc.png")