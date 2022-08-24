import torch
from levy_stable_pytorch import LevyStable
from levy_gaussian_combined import LevyGaussian
from util import score_finite_diff, gaussian_score
import matplotlib.pyplot as plt


def test_accuracy_score():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    x = torch.arange(-10, 10, 0.2)

    levy = LevyStable()
    # alpha = 2.0
    alpha = 1.7

    levy_score = levy.score(x, alpha=alpha).detach().numpy()
    levy_score_finite_diff = score_finite_diff(x, alpha=alpha)

    levy_gaussian = LevyGaussian()
    levy_score_fourier = levy_gaussian.score_cft(x, alpha=alpha,sigma_1=0, sigma_2=1).detach().numpy()
    levy_score_fft = levy_gaussian.score_fft(x, alpha=alpha,sigma_1=0, sigma_2=1).detach().numpy()

    plt.plot(x, levy_score, 'r-', lw=2, label="backpropagation")
    plt.plot(x, levy_score_finite_diff, 'y.', lw=2, label="finite diff")
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




