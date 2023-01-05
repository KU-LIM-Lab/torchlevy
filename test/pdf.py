import sys
sys.path.append("../")

from torchlevy import LevyStable, stable_dist
from scipy.stats import levy_stable

import torch
import matplotlib.pyplot as plt


def test_pdf():
    x = torch.arange(-10, 10, 0.01)
    

    for alpha in torch.arange(1.1, 2.0, 0.1):

        torch_pdf = stable_dist.pdf(x, alpha).cpu()

        scipy_pdf = stable_dist.pdf(x.cpu().numpy(), alpha=alpha.cpu().numpy(), beta=0)
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
    
    alpha = 1.7

    torch_pdf = stable_dist.pdf(x, alpha).cpu()
    scipy_pdf = stable_dist.pdf(x.cpu().numpy(), alpha=alpha, beta=0)
    scipy_pdf = torch.from_numpy(scipy_pdf)

    plt.plot(x.cpu(), torch_pdf, 'r-', label="torch_pdf")
    plt.plot(x.cpu(), scipy_pdf, 'b--', label="scipy_pdf")
    plt.ylim(0, 0.5)
    plt.title("alpha")
    plt.legend()

    plt.show()


def test_cache_pdf():
    x = torch.arange(-10, 10, 0.001)
    
    alpha = 1.7

    cache_pdf = stable_dist.pdf(x, alpha).cpu()
    non_cache_pdf = stable_dist.pdf(x, alpha, is_cache=True).cpu()

    plt.plot(x.cpu(), cache_pdf, 'r-', label="non cache pdf")
    plt.plot(x.cpu(), non_cache_pdf, 'b--', label="cache pdf")
    plt.ylim(0, 0.5)
    plt.title("alpha")
    plt.legend()

    plt.show()

def test_isotropic_pdf_less_than_1():
    """
        total volume of pdf on certain set must less than 1
         In this function, S = {x | |x| <= 1}
    """
    def n_dim_sphere_volume(n, r):
        return torch.pi ** (n/2) / torch.exp(torch.lgamma(torch.tensor(n/2 + 1))) * r ** n

    

    for alpha in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
        for dim in range(2, 100):
            is_break = False
            for r in torch.arange(0, 10, 0.5):
                x = torch.zeros(dim).reshape(1, -1)
                x[0][0] = r # x is normal unit vector

                pdf = stable_dist.pdf(x, alpha, is_isotropic=True)
                volume = n_dim_sphere_volume(n=dim, r=r)
                if volume * pdf >= 1.1:
                    print(f"when dim={dim}, alpha={alpha}, r={r}, pdf={pdf}, volume={volume} : error")
                    is_break = True
                    break
            if is_break:
                break


def test_isotropic_plot():
    

    plt.figure(figsize=(15, 6))
    for i, dim in enumerate(range(2, 300, 30)):
        plt.subplot(2, 5, i+1)
        is_break = False
        x_1d = torch.arange(0.01, 5, 0.01)
        x = torch.zeros((len(x_1d), dim))
        x[:, 0] = x_1d

        pdf = stable_dist.pdf(x, alpha=1.5, is_isotropic=True)
        plt.plot(x_1d.cpu(), pdf.cpu())
        plt.title(f"dim={dim}")
    plt.show()



if __name__ == "__main__":
    test_isotropic_plot()
