import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(1, 1)

print(list(model.parameters()))

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 2000
for epoch in range(nb_epochs+1) :
    # claculate h(x)
    prediction = model(x_train)
    # calculate cost
    cost = F.mse_loss(prediction, y_train)  # 파이토치에서 제공하는 평균 제곱 오차 함수
    
    # optimize h(x)  bt cost
    optimizer.zero_grad()   # init gradient 0
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0 :
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
        ))

new_var = torch.FloatTensor([[4.0]])
pred_y = model(new_var)
print(pred_y)   # tensor([[7.9989]], grad_fn=<AddmmBackward0>)

print(list(model.parameters()))