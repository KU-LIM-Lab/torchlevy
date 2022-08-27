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
