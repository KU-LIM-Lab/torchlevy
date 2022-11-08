import matplotlib.pyplot as plt
import torch
from torchlevy import levy_gaussian_score, LevyGaussian

def plot_two_levy_gaussian_score():
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    alpha = 1.8
    x = torch.linspace(-10, 10, 100)
    sigma1 = torch.ones(x.size()) * 0.5
    sigma2 = torch.ones(x.size()) * 0.5

    lg = LevyGaussian(alpha, sigma_1=0.5, sigma_2=0.5)
    y = lg.score(x)
    plt.plot(x.cpu(), y.cpu(), label='old')

    y = levy_gaussian_score(x, alpha=alpha, sigma1=sigma1, sigma2=sigma2)
    plt.plot(x.cpu(), y.cpu(), label="new")
    plt.legend()
    plt.grid()

    plt.subplot(122)
    x = torch.linspace(-30, 30, 100)
    lg = LevyGaussian(alpha, sigma_1=0.5, sigma_2=0.5)
    y = lg.score(x)
    plt.plot(x.cpu(), y.cpu(), label='old')

    y = levy_gaussian_score(x, alpha=alpha, sigma1=sigma1, sigma2=sigma2)
    plt.plot(x.cpu(), y.cpu(), label="new")
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig("tmp.png")

if __name__ == "__main__":
    plot_two_levy_gaussian_score()