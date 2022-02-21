import torch
import torch.nn as nn

inputs = torch.Tensor(1, 1, 28, 28) # torch.Size([1, 1, 28, 28])

conv1 = nn.Conv2d(1, 32, 3, padding=1)

conv2 = nn.Conv2d(32, 64, 3, padding=1)

pool = nn.MaxPool2d(2)

out = conv1(inputs)
out = pool(out)
out = conv2(out)
out = pool(out)

# 첫 차원인 배치 차원은 그대로 두고 나머지를 펼치기
out = out.view(out.size(0), -1)

fc = nn.Linear(3136, 10)
out = fc(out)
print(out.shape)  # torch.Size([1, 10])