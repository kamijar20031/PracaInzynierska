import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/nonexistent"
import tensorflow as tf
tf.config.optimizer.set_jit(False)
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.config.list_physical_devices('GPU'))

keras.utils.set_random_seed(42)
# Pobieranie danych
df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/heart.csv")
categorical_variables = ['sex', 'cp', 'fbs','restecg', 'exang', 'ca', 'thal']
numerics = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']

# One Hot Encodowanie danych - kazda kategoria jest rozdzielana na jej dzieci i tam sa wstawiane 0 lub 1
df = pd.get_dummies(df, columns = categorical_variables)
# Rozdzielenie danych
test_df = df.sample(frac=0.2, random_state=42)
train_df = df.drop(test_df.index)
# Normalizacja
means = train_df[numerics].mean()
sd = train_df[numerics].std()
train_df[numerics] = (train_df[numerics]- means)/sd
test_df[numerics] = (test_df[numerics]-means)/sd

# Konwersja danych na tablice numpy
train = train_df.to_numpy(dtype=np.float32)
test = test_df.to_numpy(dtype=np.float32)

# Wybieramy tylko dane bez target z kopii danych - czyli bez indeksu 6
train_X = np.delete(train, 6, axis=1)
test_X = np.delete(test, 6, axis=1)

# Tutaj z kolei wybieramy sam target
train_Y = train[:,6]
test_Y = test[:,6]

# Ilosc kategorii
num_columns = train_X.shape[1]

# Wejscie dla modelu
input = keras.Input(shape=(num_columns,))

# "Karmimy" pojedyncza schowana warstwe wejsciem jaki mielismy. Ustawiamy funkcje aktywacji "relu", poniewaz domyslnie zawsze jest uzywana dla gestych ukrytych warstw. Mozna tez ustawic nazwe warstwy
h = keras.layers.Dense(16, activation="relu", name="Hidden")(input)

# Wyjscie ukrytej warswy wsadzamy do nastepnej - ta juz wskazuje prawdopodobienstwa, wiec musi uzywac sigmoid
output = keras.layers.Dense(1, activation="sigmoid", name="Output")(h)

# Ustawiamy model jako mix naszego inputu i outputu!
model = keras.Model(input, output)


# Model: "model"
#________________________________________________________
# Layer (type)                Output Shape       Param #   
#========================================================
# input_1 (InputLayer)        [(None, 29)]         0         
#                                                            #     
# Hidden (Dense)              (None, 16)          480       
#                                                            #     
# Output (Dense)              (None, 1)            17        
#                                                            #     
#=========================================================
#Total params: 497 (1.94 KB)
#Trainable params: 497 (1.94 KB)
#Non-trainable params: 0 (0.00 Byte)

# Ilosc parametrow oblicza sie nastepujaco:
# (input + 1)*hidden + (hidden + 1)*output

# Kompilacja modelu
# Wybierany jest optymizator (zwykle adam), mowi jak zmieniac wagi modelu, np Stochastic Gradient Descent
# Wybieramy funkcje straty, dla klasyfikacji binarnej binary_crossentropy, a np dla regresji liniowej dajemy mse
# Wybieramy tez to na co mamy zwracac uwage przy optymizatorze
model.compile(optimizer="adam", loss="binary_crossentropy", metrics =["accuracy"])

# Teraz juz fitujemy model uzywajac danych, robimy 300 powtorzen a dane wrzucamy w pojemnikach 32 - dla kazdego z nich model wykonuje obliczenia
# Optimizer dziala dla kazdego batcha - w kazdym epochu moze byc robione wiecej niz jeden optimization!
# Verbosity mowi o tym czy widzimy logi
# Validation split jeszcze dodatkowo rodziela train_data, mamy wtedy liczone val_accuracy i val_loss
history = model.fit(train_X, train_Y, epochs=300, batch_size=32, verbose = 1, validation_split=0.2)
# W koncu badane sa dane testowe!
print(model.evaluate(test_X, test_Y))

hisdict = history.history
lossval = hisdict["loss"]
vallosval = hisdict["val_loss"]
epochs = range(1,len(lossval)+1)
plt.figure()
plt.plot(epochs, lossval, "bo", label = "Training loss")
plt.plot(epochs, vallosval, "b", label="Validation loss")
plt.title("Traning and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")

plt.clf()
acc = hisdict["accuracy"]
valacc = hisdict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, valacc, "b",label="Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("acc.png")
