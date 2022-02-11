import torch

w = torch.tensor(2.0, requires_grad=True)

y = w**2
z = 2*y + 5

z.backward()

print('w로 미분한 값 {}' .format(w.grad))  # w로 미분한 값 8.0