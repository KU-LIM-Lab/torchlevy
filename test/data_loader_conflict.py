import sys
sys.path.append("../")

import torch
torch.multiprocessing.set_start_method('spawn')

from torchvision import datasets
from torchvision.transforms import ToTensor
from torchlevy import LevyStable, stable_dist
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def test_data_loader_conflict():

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)

    for (i, x) in enumerate(train_loader):

        
        x = stable_dist.sample(alpha=2, size=1000)
        assert(not str(x.device) == "cpu")

        if i == 100:
            break


def test_data_loader_conflict2(num_workers=0):

    training_data = datasets.FashionMNIST('data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(training_data, batch_size=64,
                         shuffle=True, num_workers=num_workers, generator=torch.Generator(device='cuda'))

    for (i, x) in enumerate(train_loader):

        
        x = stable_dist.sample(alpha=2, size=1000)
        assert(not str(x.device) == "cpu")

        if i == 100:
            break


if __name__ == "__main__":
    test_data_loader_conflict()