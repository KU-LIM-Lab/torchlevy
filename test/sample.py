from scipy.stats import levy_stable
import random
from torchlevy import LevyStable
import matplotlib.pyplot as plt
import numpy as np
import torch


def test_sample_clamp():
    for clamp in range(2, 100):
        levy = LevyStable()
        y2 = levy.sample(alpha=1.7, size=(100, 100, 100), clamp=clamp)
        assert (not torch.any(y2 > clamp))


def compare_scipy_and_torch():
    alphas = [1.7]  # [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    beta = 1

    for alpha in alphas:
        scale = random.uniform(1, 10)

        scipy_sample = levy_stable.rvs(alpha, beta, loc=0., scale=scale, size=100000)
        scipy_sample = np.clip(scipy_sample, -100, 100)

        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.hist(scipy_sample, bins=2000, facecolor='y', alpha=0.5, label="scipy_sample")
        plt.xlim(-50, 100)
        plt.legend()

        levy = LevyStable()
        torch_sample = levy.sample(alpha, beta, loc=0., scale=scale, size=100000).cpu()
        torch_sample = torch.clip(torch_sample, -100, 100)
        plt.subplot(122)
        plt.hist(torch_sample, bins=2000, facecolor='blue', alpha=0.5, label="torch_sample")
        plt.xlim(-50, 100)
        plt.legend()
        plt.show()


def test_isotropic():
    levy = LevyStable()

    alpha = 1.7
    e = levy.sample(alpha, size=[10000, 2], is_isotropic=True, reject_threshold=40).cpu()
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plt.scatter(e[:, 0], e[:, 1], marker='.')
    plt.gca().set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    # test_sample_clamp()
    # compare_scipy_and_torch()
    test_isotropic()
