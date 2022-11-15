import matplotlib.pyplot as plt
import torch
from torchlevy import LevyGaussian
from torchlevy.levy_gaussian import levy_gaussian_score, LevyGaussian

def plot_levy_gaussian_score():
    plt.figure(figsize=(12,4))

    alphas = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    for i, alpha in enumerate(alphas):
        plt.subplot(2, 5, i+1)
        x = torch.linspace(-10, 10, 100)

        lg = LevyGaussian(alpha, sigma_1=0., sigma_2=1.)
        y = lg.score(x)
        plt.plot(x.cpu(), y.cpu())
        plt.title(f"alpha={alpha}")
        plt.gca().set_aspect('equal')
        plt.grid(True)
    plt.show()


def test():
    alpha = 1.7
    t = 1000
    x = torch.randn(t, 100, 100, 3)
    sigma1s = torch.linspace(0, 1, t)
    sigma2s = torch.linspace(1, 0, t)

    ret = levy_gaussian_score(alpha, x, sigma1s, sigma2s)




if __name__ == "__main__":
    test()