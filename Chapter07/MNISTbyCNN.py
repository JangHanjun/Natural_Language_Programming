import torch
import torchvision.datasets as dsets
import torchvidion.transforms as transforms
import torch.nn.init

torch.manual_seed(777)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    