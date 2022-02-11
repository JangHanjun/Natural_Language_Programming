import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# fix random seed
torch.manual_seed(1)

# Data
x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

# weight & bais
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
        # calculate H(x)
        hypothesis = x_train.matmul(W) + b
        # calculate cost
        cost = torch.mean((hypothesis - y_train) ** 2)
        
        # optimizer H(x) by cost
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # print log 
        print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
            epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
        ))