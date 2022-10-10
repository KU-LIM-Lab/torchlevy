import sys
sys.path.append("../")

from torchlevy import LevyStable
from scipy.stats import levy_stable

import torch
import matplotlib.pyplot as plt


def test_pdf():
    x = torch.arange(-10, 10, 0.01)
    levy = LevyStable()

    for alpha in torch.arange(1.1, 2.0, 0.1):

        torch_pdf = levy.pdf(x, alpha).cpu()

        scipy_pdf = levy_stable.pdf(x.cpu().numpy(), alpha=alpha.cpu().numpy(), beta=0)
        scipy_pdf = torch.from_numpy(scipy_pdf)

        diff = torch.abs(torch_pdf - scipy_pdf)
        if torch.any(diff > 1e-4):
            indices = x[torch.any(diff > 1e-4)]
            print(scipy_pdf[indices])
            print(scipy_pdf[indices])
            print()
            raise RuntimeError

def test_plot_pdf():
    x = torch.arange(-10, 10, 0.001)
    levy = LevyStable()
    alpha = 1.7

    torch_pdf = levy.pdf(x, alpha).cpu()
    scipy_pdf = levy_stable.pdf(x.cpu().numpy(), alpha=alpha, beta=0)
    scipy_pdf = torch.from_numpy(scipy_pdf)

    plt.plot(x.cpu(), torch_pdf, 'r-', label="torch_pdf")
    plt.plot(x.cpu(), scipy_pdf, 'b--', label="scipy_pdf")
    plt.ylim(0, 0.5)
    plt.title("alpha")
    plt.legend()

    plt.show()


def test_cache_pdf():
    x = torch.arange(-10, 10, 0.001)
    levy = LevyStable()
    alpha = 1.7

    cache_pdf = levy.pdf(x, alpha).cpu()
    non_cache_pdf = levy.pdf(x, alpha, is_cache=True).cpu()

    plt.plot(x.cpu(), cache_pdf, 'r-', label="non cache pdf")
    plt.plot(x.cpu(), non_cache_pdf, 'b--', label="cache pdf")
    plt.ylim(0, 0.5)
    plt.title("alpha")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    test_cache_pdf()
