import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Tworzymy elementy, cov to [varX, covX/Y, covY/X, varY]

samples = 1000
neg = np.random.multivariate_normal(mean=[0, 3], cov = [[1,0.5],[0.5,1]], size=samples)
pos = np.random.multivariate_normal(mean=[3, 0], cov = [[1,0.5],[0.5,1]], size=samples)

# vstack uklada obie macierze w tensor 2000x1x2
inputs = np.vstack((neg, pos)).astype(np.float32)
targets = np.vstack((np.zeros((samples, 1), dtype="float32"), np.ones((samples, 1), dtype="float32")))

# Rysowanie ukladu
plt.scatter(inputs[:, 0], inputs[:,1], c=targets[:,0])
plt.savefig("plot.png")

# Ustawianie wektora poczatkowego W i b, gdzie y = Wx + b
W = tf.Variable(initial_value=tf.random.uniform(shape=(2,1)))
b = tf.Variable(initial_value=tf.zeros(shape=(1,)))

# Ustawienie wyniku modelu: regresja liniowa y = Wx + b
def model(inputs, W, b):
    return tf.matmul(inputs, W) + b

# Obliczanie bledu miedzy target a predict - otrzymujemy wektor a zwracamy skalar
def mean_sqr_err(targets, predicts):
    losses = tf.square(targets-predicts)
    return tf.reduce_mean(losses)

learning_rate = 0.01

# Wykonywanie jednego kroku obliczenia - obliczamy predykcje, roznice miedzy nimi a rzeczywistymi wynikami i w oparciu o gradient funkcji loss wzgledem W i b szukamy najbardziej optymalnego wyniku (miejsce zerowe gradientu, czyli najmniejszy loss). W i b potem aktualizujemy.
@tf.function(jit_compile=True)
def training_step(inputs, targets, W, b):
    with tf.GradientTape() as tape:
        predictions = model(inputs, W, b)
        loss = mean_sqr_err(targets, predictions)
    grad_W, grad_B = tape.gradient(loss, [W, b])
    W.assign_sub(grad_W* learning_rate)
    b.assign_sub(grad_B*learning_rate)
    return loss

# Wykonanie 400 krokow obliczeniowych
for step in range(400):
    loss = training_step(inputs, targets, W, b)
    print(f"Loss at step {step}: {loss:.4f}")

# Wykonanie ostatecznej predykcji i naniesienie wynikow na wykres
prediction = model(inputs, W, b)
x = np.linspace(-1, 4,100)
# Normalne rownianie ma postac:
# f(x,y) = x*W[0] +y*W[1] + b, a to co otrzymujemy to rzeksztalcenie przy podstawieniu f(x,y) = 0.5
y = -W[0]/ W[1] * x + (0.5-b)/W[1]
plt.plot(x,y,"-r")
plt.scatter(inputs[:,0], inputs[:,1], c=prediction[:,0] > 0.5)
plt.savefig("plotRes.png")