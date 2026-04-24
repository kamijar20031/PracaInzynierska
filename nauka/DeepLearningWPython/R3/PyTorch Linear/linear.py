import torch
import numpy as np
import matplotlib.pyplot as plt
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

learning_rate = 0.01

def mean_sqr_err(targets, predicts):
    loss = torch.square(targets-predicts)
    return torch.mean(loss)

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.rand(2,1))
        self.b = torch.nn.Parameter(torch.zeros(1))
    def forward(self, inputs):
        return torch.matmul(inputs, self.W) + self.b

model = LinearModel()
inputs = torch.tensor(inputs)
targets = torch.tensor(targets)
optimizer = torch.optim.SGD(model.parameters(), lr =learning_rate)

def training_step(inputs, targets):
    preds = model(inputs)
    loss = mean_sqr_err(targets, preds)
    loss.backward()
    optimizer.step()
    model.zero_grad()
    return loss

for step in range(400):
    loss = training_step(inputs, targets)
    print(f"Loss at step {step}: {loss:.4f}")

# Wykonanie ostatecznej predykcji i naniesienie wynikow na wykres
prediction = model(inputs)
plt.scatter(inputs[:,0], inputs[:,1], c=prediction[:,0] > 0.5)
plt.savefig("plotRes.png")