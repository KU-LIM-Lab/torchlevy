import sys
sys.path.append("../")

import torch
from torchlevy import LevyGaussian
import time


@profile
def test_memory():
    x = torch.randn(size=(3, 200, 200))
    for epoch in range(10):
        start = time.time()
        for i in range(100):
            levy_gaussian = LevyGaussian(alpha=1.7, sigma_1=1-i/1000, sigma_2=i/1000)
            score = levy_gaussian.score(x)
        print(f"epoch {epoch} :", time.time() - start)


@profile
def test_memory2():
    x = torch.randn(size=(1,), device='cpu')
    print(x.device)


if __name__ == "__main__":
    test_memory()