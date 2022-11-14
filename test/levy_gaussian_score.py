import matplotlib.pyplot as plt
import torch
from torchlevy import LevyGaussian

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



if __name__ == "__main__":
    plot_levy_gaussian_score()