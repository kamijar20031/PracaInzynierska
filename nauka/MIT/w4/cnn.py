import tensorflow as tf
from tensorflow import keras
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

def plot_loss_curves(history):
    plt.clf()
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png")

def plot_acc_curves(history):
    plt.clf()
    history_dict = history.history
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("acc.png")

x_train = x_train/255.0
x_test = x_test/255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

input = keras.Input(shape=x_train.shape[1:])
x = keras.layers.Conv2D(32, kernel_size=(2,2), activation="relu", name="Conv_1")(input)
x = keras.layers.MaxPool2D()(x)

x = keras.layers.Conv2D(32, kernel_size=(2,2), activation="relu", name="Conv_2")(input)
x = keras.layers.MaxPool2D()(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256,activation="relu")(x)

output = keras.layers.Dense(10, activation="softmax")(x)
model = keras.Model(input, output)

# Model: "model"
# _________________________________________________________________
#  Layer (type)                   Output Shape              Param #   
# =================================================================
#  input_1 (InputLayer)           [(None, 28, 28, 1)]       0         
#                                                                  
#  Conv_1 (Conv2D)                (None, 27, 27, 32)        160       
#                                                                  
#  max_pooling2d (MaxPooling2D)   (None, 13, 13, 32)        0                                                                       
#                                                                   
#  Conv_2 (Conv2D)                (None, 12, 12, 32)        4128      
#                                                                  
#  max_pooling2d1 (MaxPooling2D)  (None, 6, 6, 32)          0                                                                     
#                                                                 
#  flatten (Flatten)              (None, 1152)              0         
#                                                                  
#  dense (Dense)                  (None, 256)               295168    
#                                                                 
#  dense_1 (Dense)                (None, 10)                2570      
#                                                                 
# =================================================================
#  Total params: 302026 (1.15 MB)
#  Trainable params: 302026 (1.15 MB)
#  Non-trainable params: 0 (0.00 Byte)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

plot_loss_curves(history)
plot_acc_curves(history)

score = model.evaluate(x_test, y_test)
print("Test accuracy:", score[1])
