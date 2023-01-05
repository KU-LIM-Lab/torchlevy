from scipy.stats import levy_stable
import random
from torchlevy import LevyStable, stable_dist
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns


def test_rejection_sampling():
    for threshold in range(2, 100):
        
        y2 = stable_dist.sample(alpha=1.7, size=(100, 100, 100), reject_threshold=threshold)
        assert (not torch.any(y2 > threshold))


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

        torch_sample = stable_dist.sample(alpha, beta, loc=0., scale=scale, size=100000).cpu()
        torch_sample = torch.clip(torch_sample, -100, 100)
        plt.subplot(122)
        plt.hist(torch_sample, bins=2000, facecolor='blue', alpha=0.5, label="torch_sample")
        plt.xlim(-50, 100)
        plt.legend()
        plt.show()


def plot_isotropic():

    alpha = 1.5

    plt.figure(figsize=(12, 4))
    plt.rcParams['font.size'] = 18
    plt.subplot(131)

    gaussian_noise = stable_dist.sample(alpha=2.0, size=[10000, 2]).cpu()
    sns.scatterplot(x=gaussian_noise[:, 0], y=gaussian_noise[:, 1])
    plt.gca().set_aspect('equal')
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.title(r"Gaussian")
    plt.xticks([-20, 0, 20])
    plt.yticks([])


    plt.subplot(132)
    non_isotropic_noise = stable_dist.sample(alpha, size=[10000, 2], is_isotropic=False).cpu()
    sns.scatterplot(x=non_isotropic_noise[:, 0], y=non_isotropic_noise[:, 1])
    plt.gca().set_aspect('equal')
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.title(r"Independent $\alpha$=1.5")
    plt.xticks([-20, 0, 20])
    plt.yticks([])

    plt.subplot(133)
    isotropic_noise = stable_dist.sample(alpha, size=[10000, 2], is_isotropic=True).cpu()
    sns.scatterplot(x=isotropic_noise[:, 0], y=isotropic_noise[:, 1])
    plt.gca().set_aspect('equal')
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.title(r"Isotropic $\alpha$=1.5")
    plt.xticks([-20, 0, 20])
    plt.yticks([])
    plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05)

    plt.savefig("three_different_samples.pdf")

    plt.show()

def test_isotropic():

    
    alpha = 1.8
    e = stable_dist.sample(alpha, size=[100, 3, 128, 128], is_isotropic=False).cpu()
    print("isotropic: ", e.max(), e.min())
    e = stable_dist.sample(alpha, size=[100, 3, 128, 128], is_isotropic=True).cpu()
    print("non-isotropic: ", e.max(), e.min())

def test_nan():

    
    alphas = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    # for alpha in alphas:
    #     for i in range(1000):
    #         e = levy_stable.sample(alpha, size=(100, 100, 100))
    #         assert torch.all(torch.isfinite(e))
    #
    #         e = levy_stable.sample(alpha, size=(100, 100, 100), is_isotropic=True, clamp_threshold=20)
    #         assert torch.all(torch.isfinite(e))
    for i in range(1000):
        e = stable_dist.sample(1.2, size=(50000, 2), is_isotropic=True)
        assert not torch.any(torch.isnan(e))
    print("test nan passed")


def test_beta1():
    

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for alpha in alphas:
        for i in range(1000):
            x = stable_dist.sample(alpha, beta=1, size=(50000,))
            if torch.any(x < 0):
                print(alpha)
                print(1111, x[x < 0])
                # raise RuntimeError()


def test_scipy_beta1():
    # 

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for alpha in alphas:
        for i in range(1000):
            x = levy_stable.rvs(alpha, beta=1, size=50000)
            if torch.any(torch.from_numpy(x < 0)):
                print(alpha)
                print(1111, x[x < 0])


if __name__ == "__main__":
    # test_sample_clamp()
    # compare_scipy_and_torch()
    # plot_isotropic()
    plot_isotropic()

