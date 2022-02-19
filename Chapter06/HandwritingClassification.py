import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import torch
import torch.nn as nn
from torch import optim

digits = load_digits()

X = digits.data
Y = digits.target
model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
    )
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

loss_fn = nn.CrossEntropyLoss() # include softmax function
optimizer = optim.Adam(model.parameters())
losses = []

for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print('Epoch {:4d}/{}  Cost: {:.6f}'.format(epoch, 100, loss.item()))
    
    losses.append(loss.item())

plt.plot(losses)