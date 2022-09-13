import sys
sys.path.append("../")

import torch
from torchlevy import LevyGaussian
import time


def test_cache_improvement():
    alpha = 1.7
    x = torch.randn((3, 32, 32))
    # x = torch.randn((3, 128, 128))

    start = time.time()
    levy_gaussian = LevyGaussian(alpha=alpha, sigma_1=1, sigma_2=1, type="cft")
    tmp = levy_gaussian.score(x)
    print("\n")
    print(f"first computation takes {time.time() - start}s")

    start = time.time()
    levy_gaussian = LevyGaussian(alpha=alpha, sigma_1=1, sigma_2=1, type="cft")
    tmp = levy_gaussian.score(x)
    print(f"second computation takes {time.time() - start}s")

    start = time.time()
    levy_gaussian = LevyGaussian(alpha=alpha, sigma_1=1, sigma_2=1, type="cft")
    tmp = levy_gaussian.score(x)
    print(f"third computation takes {time.time() - start}s")


def test_cache_improvement2():
    for epoch in range(10):
        start = time.time()
        for i in range(1000):
            x = torch.randn(size=(3, 200, 200))
            levy_gaussian = LevyGaussian(alpha=1.7, sigma_1=1-i/1000, sigma_2=i/1000)
            score = levy_gaussian.score(x)
        print(f"epoch {epoch} :", time.time() - start)



if __name__ == "__main__":
    # test_cache_improvement()
    test_cache_improvement2()