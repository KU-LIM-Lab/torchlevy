import sys

sys.path.append("../")

import torch
from torchlevy import LevyStable, levy_stable
from torchlevy import LevyGaussian
from torchlevy.util import score_finite_diff, gaussian_score
import matplotlib.pyplot as plt
import numpy as np


def test_score_methods_plot(alpha=1.5, x=torch.arange(-100, 100, 0.5)):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    

    levy_score_default = levy_stable.score(x, alpha=alpha, type="default").detach().cpu().numpy()
    levy_score_backprop = levy_stable.score(x, alpha=alpha, type="backpropagation").detach().cpu().numpy()
    levy_score_finite_diff = score_finite_diff(x.cpu().numpy(), alpha=alpha)

    levy_gaussian = LevyGaussian(alpha=alpha, sigma_1=0, sigma_2=1, type='cft')
    levy_score_fourier = levy_gaussian.score(x).detach().cpu().numpy()

    levy_gaussian = LevyGaussian(alpha=alpha, sigma_1=0, sigma_2=1, type='fft')
    levy_score_fft = levy_gaussian.score(x).detach().cpu().numpy()

    x = x.cpu()

    plt.figure(figsize=(15, 4))
    plt.subplot(151)
    plt.plot(x, levy_score_default, 'c-', lw=2, label="default")
    plt.legend()
    plt.subplot(152)
    plt.plot(x, levy_score_backprop, 'r-', lw=2, label="backpropagation")
    plt.legend()
    plt.subplot(153)
    plt.plot(x, levy_score_finite_diff, 'y-', lw=2, label="finite diff")
    plt.legend()
    plt.subplot(154)
    plt.plot(x, levy_score_fourier, 'b-', lw=2, label="cft")
    plt.legend()
    plt.subplot(155)
    plt.plot(x, levy_score_fft, 'g-', lw=2, label="fft")
    plt.legend()
    plt.show()

    plt.plot(x, levy_score_default, 'c-', lw=2, label="default")
    plt.plot(x, levy_score_backprop, 'r-', lw=2, label="backpropagation")
    plt.plot(x, levy_score_finite_diff, 'y-', lw=2, label="finite diff")
    plt.plot(x, levy_score_fourier, 'b.', lw=2, label="cft")
    plt.plot(x, levy_score_fft, 'g.', lw=2, label="fft")
    if alpha == 2:
        plt.plot(x, gaussian_score(x), label='ground truth')
        print("\n")
        print("levy_score               :", torch.sum((gaussian_score(x) - levy_score_backprop) ** 2))
        print("levy_score_finite_diff   :", torch.sum((gaussian_score(x) - levy_score_finite_diff) ** 2))
        print("levy_score_fourier       :", torch.sum((gaussian_score(x) - levy_score_fourier) ** 2))
        print("levy_score_fft           :", torch.sum((gaussian_score(x) - levy_score_fft) ** 2))

    plt.legend()
    plt.show()


def test_score_diff_plot(alpha=1.5, x=torch.arange(-15, 15, 0.3)):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    
    levy_score = levy_stable.score(x, alpha=alpha, type="backpropagation").detach().cpu().numpy()
    levy_score_default = levy_stable.score(x, alpha=alpha, type="default").detach().cpu().numpy()

    levy_gaussian = LevyGaussian(alpha=alpha, sigma_1=0, sigma_2=1)
    levy_score_cft = levy_gaussian.score(x).detach().cpu().numpy()

    default_diff = np.abs((levy_score_default - levy_score_cft))
    cft_diff = np.abs((levy_score - levy_gaussian))

    plt.figure(figsize=(8, 3))

    x = x.cpu()
    plt.subplot(121)
    plt.plot(x, levy_score, 'r-', lw=2, label="backpropagation")
    plt.plot(x, levy_score_default, 'c.', lw=2, label="default")
    plt.plot(x, levy_score_cft, 'b.', lw=2, label="cft")
    plt.legend()

    plt.subplot(122)
    plt.plot(x, cft_diff, 'r-', lw=2, label="diff between cft and backpro")
    plt.plot(x, default_diff, 'c-', lw=2, label="diff between default and backpro")
    plt.ylim((0, 0.005))
    plt.legend()

    plt.show()


def test_nan():
    alphas = [1.5, 1.7, 2.0]
    x = torch.arange(-1000, 1000, 0.1)

    for alpha in alphas:
        
        levy_score = levy_stable.score(x, alpha=alpha)
        if torch.any(levy_score.isnan()):
            raise RuntimeError(f"levy_socre has nan at alpha={alpha}")

        levy_gaussian = LevyGaussian(alpha=alpha, sigma_1=0, sigma_2=1)
        levy_score_cft = levy_gaussian.score(x)
        if torch.any(levy_score_cft.isnan()):
            raise RuntimeError(f"levy_score_cft has nan at alpha={alpha}")

        levy_gaussian = LevyGaussian(alpha=alpha, sigma_1=0, sigma_2=1)
        levy_score_fft = levy_gaussian.score(x)
        if torch.any(levy_score_fft.isnan()):
            raise RuntimeError(f"levy_score_fft has nan at alpha={alpha}")

def test_score(alpha=1.5):
    
    x = torch.arange(-5, 5, 0.01)
    score = levy_stable.score(x, alpha, type="cft", is_fdsm=True)

    plt.plot(x.cpu(), score.cpu(), '-')
    plt.ylim(-3, 3)
    plt.show()

def test_isotropic_score(alpha=1.5):
    
    # x = torch.randn(100, 3, 32, 24)
    x = torch.arange(-30, 30, 0.5)[:, None]  # [:, None, None, None]
    n = len(x)

    plt.figure(figsize=(12, 4))
    for dim in range(2, 6):
        plt.subplot(1, 4, dim-1)
        if dim != 1:
            x = torch.cat([x, torch.zeros(n, 1)], dim=1)
        # if dim == 5:
        levy_score = levy_stable.score(x, alpha=alpha, is_isotropic=True)
        print(f"dim={dim} :", torch.abs(x / alpha + levy_score).mean().item())

        plt.plot(x[:, 0].cpu(), levy_score[:, 0].cpu(), '-')
        plt.title(f"dim={dim}")
        plt.ylim(-20, 20)
    plt.suptitle(f"alpha={alpha}")
    plt.tight_layout(pad=0.8)
    plt.show()


if __name__ == "__main__":
    # test_nan()
    # test_score()
    test_isotropic_score()
    # test_origin_score_is_dim1_isotropic_score()
