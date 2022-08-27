import torch
from levy_stable_pytorch.levy import LevyStable
from levy_stable_pytorch.levy_gaussian import LevyGaussian
from levy_stable_pytorch.util import score_finite_diff, gaussian_score
import matplotlib.pyplot as plt
import numpy as np


def test_score_methods_plot(alpha=1.7, x=torch.arange(-15, 15, 0.1)):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    levy = LevyStable()

    levy_score = levy.score(x, alpha=alpha).detach().numpy()
    levy_score_finite_diff = score_finite_diff(x, alpha=alpha)

    levy_gaussian = LevyGaussian(alpha=alpha,sigma_1=0, sigma_2=1, type='cft')
    levy_score_fourier = levy_gaussian.score(x).detach().numpy()

    levy_gaussian = LevyGaussian(alpha=alpha,sigma_1=0, sigma_2=1, type='fft')
    levy_score_fft = levy_gaussian.score(x).detach().numpy()


    plt.figure(figsize=(15, 4))
    plt.subplot(141)
    plt.plot(x, levy_score, 'r-', lw=2, label="backpropagation")
    plt.legend()
    plt.subplot(142)
    plt.plot(x, levy_score_finite_diff, 'y-', lw=2, label="finite diff")
    plt.legend()
    plt.subplot(143)
    plt.plot(x, levy_score_fourier, 'b-', lw=2, label="cft")
    plt.legend()
    plt.subplot(144)
    plt.plot(x, levy_score_fft, 'g-', lw=2, label="fft")
    plt.legend()
    plt.show()

    plt.plot(x, levy_score, 'r-', lw=2, label="backpropagation")
    plt.plot(x, levy_score_finite_diff, 'y-', lw=2, label="finite diff")
    plt.plot(x, levy_score_fourier, 'b.', lw=2, label="cft")
    plt.plot(x, levy_score_fft, 'g.', lw=2, label="fft")
    if alpha==2:
        plt.plot(x, gaussian_score(x), label='ground truth')
        print("\n")
        print("levy_score               :", torch.sum((gaussian_score(x) - levy_score) ** 2))
        print("levy_score_finite_diff   :", torch.sum((gaussian_score(x) - levy_score_finite_diff) ** 2))
        print("levy_score_fourier       :", torch.sum((gaussian_score(x) - levy_score_fourier) ** 2))
        print("levy_score_fft           :", torch.sum((gaussian_score(x) - levy_score_fft) ** 2))

    plt.legend()
    plt.show()


def test_score_diff_plot(alpha=1.7, x=torch.arange(-15, 15, 0.1)):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)


    levy = LevyStable()
    levy_score = levy.score(x, alpha=alpha).detach().numpy()

    levy_gaussian = LevyGaussian(alpha=alpha,sigma_1=0, sigma_2=1, type='cft')
    levy_score_cft = levy_gaussian.score(x).detach().numpy()

    levy_gaussian = LevyGaussian(alpha=alpha,sigma_1=0, sigma_2=1, type='fft')
    levy_score_fft = levy_gaussian.score(x).detach().numpy()

    cft_diff = np.abs((levy_score - levy_score_cft))
    fft_diff = np.abs((levy_score - levy_score_fft))

    plt.figure(figsize=(12, 3))

    plt.subplot(131)
    plt.plot(x, levy_score, 'r-', lw=2, label="backpropagation")
    plt.plot(x, levy_score_cft, 'b.', lw=2, label="cft")
    plt.plot(x, levy_score_fft, 'g.', lw=2, label="fft")
    plt.legend()

    plt.subplot(132)
    plt.plot(x, cft_diff, 'r-', lw=2, label="diff between cft and backpro")
    plt.plot(x, fft_diff, 'y-', lw=2, label="diff between fft and backpro")
    plt.ylim((0, 0.01))
    plt.legend()

    plt.subplot(133)
    plt.plot(x, cft_diff, 'r-', lw=2, label="diff between cft and backpro")
    plt.plot(x, fft_diff, 'y-', lw=2, label="diff between fft and backpro")
    plt.ylim((0, 0.05))
    plt.legend()

    plt.show()

