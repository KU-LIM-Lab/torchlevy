import sys
sys.path.append("../")

from torchlevy import LevyGaussian
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

from torch.utils.data import DataLoader

train_loader = DataLoader(training_data, batch_size=64, shuffle=True)

for i, x in enumerate(train_loader):
    break