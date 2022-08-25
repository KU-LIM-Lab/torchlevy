import torch
from levy_stable_pytorch import LevyStable
from levy_gaussian_combined import LevyGaussian
from util import score_finite_diff, gaussian_score
import matplotlib.pyplot as plt
import numpy as np

def test_pdf_sample_score_plot():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    range_ = 10

    x = torch.arange(-range_, range_, 0.1)

    levy = LevyStable()
    # alpha = 2.0
    alpha = 1.7

    pdf = levy.pdf(x, alpha)
    sample = levy.sample(alpha, size=10000).numpy()
    score = levy.score(x, alpha).detach().numpy()

    plt.figure(figsize=(12, 3))

    plt.subplot(131)
    plt.plot(x, pdf, 'r-', lw=2, label="pdf")
    plt.xlim((-range_, range_))
    plt.ylim((0, 0.3))
    plt.legend()


    plt.subplot(132)

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    # plt.hist(sample, label="sample", alpha=0.5)
    plt.hist(sample, 2000, facecolor='blue', alpha=0.5)
    plt.xlim((-range_, range_))
    plt.legend()

    plt.subplot(133)
    plt.plot(x, score, 'g-', lw=2, label="score")
    plt.xlim((-range_, range_))
    plt.legend()

    plt.show()

def test_score_compare():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x = torch.arange(-100, 100, 1.)

    levy = LevyStable()
    # alpha = 2.0
    alpha = 1.7

    levy_score = levy.score(x, alpha=alpha).detach().numpy()
    levy_score_finite_diff = score_finite_diff(x, alpha=alpha)

    levy_gaussian = LevyGaussian(alpha=alpha,sigma_1=0, sigma_2=1, type='cft')
    levy_score_fourier = levy_gaussian.score(x).detach().numpy()

    levy_gaussian = LevyGaussian(alpha=alpha,sigma_1=0, sigma_2=1, type='fft')
    levy_score_fft = levy_gaussian.score(x).detach().numpy()

    plt.plot(x, levy_score, 'r-', lw=2, label="backpropagation")
    plt.plot(x, levy_score_finite_diff, 'y-', lw=2, label="finite diff")
    plt.plot(x, levy_score_fourier, 'b.', lw=2, label="fourier transform")
    plt.plot(x, levy_score_fft, 'g.', lw=2, label="fast fourier transform")
    if alpha==2:
        plt.plot(x, gaussian_score(x), label='ground truth')
        print("\n")
        print("levy_score               :", torch.sum((gaussian_score(x) - levy_score) ** 2))
        print("levy_score_finite_diff   :", torch.sum((gaussian_score(x) - levy_score_finite_diff) ** 2))
        print("levy_score_fourier       :", torch.sum((gaussian_score(x) - levy_score_fourier) ** 2))
        print("levy_score_fft           :", torch.sum((gaussian_score(x) - levy_score_fft) ** 2))

    plt.legend()
    plt.show()


def test_score_diff():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    x = torch.arange(-100, 100, 0.6)

    levy = LevyStable()
    # alpha = 2.0
    alpha = 1.7

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
    plt.plot(x, levy_score_cft, 'b.', lw=2, label="fourier transform")
    plt.plot(x, levy_score_fft, 'g.', lw=2, label="fast fourier transform")
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


