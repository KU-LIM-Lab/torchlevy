import torch
from torchquad import Simpson  # The available integrators
from torchlevy import LevyStable

def levy_gaussian_score(x :torch.Tensor, alpha, sigma1, sigma2):
    def cft(g1, g2, x):
        """Numerically evaluate the Fourier Transform of g for the given frequencies"""
        simp = Simpson()
        score = simp.integrate(lambda y: g1(x - y) * g2(y), dim=1, N=999, integration_domain=[[-15, 15]])
        if torch.any(torch.abs(x) > 8):
            score_large_domain = simp.integrate(lambda y: g1(x - y) * g2(y), dim=1, N=999, integration_domain=[[-50, 50]])
            score[torch.abs(x) > 8] = score_large_domain[torch.abs(x) > 8]

        return score

    def gaussian_pdf(x, mu=0, sigma=1):
        return 1 / (sigma * torch.sqrt(torch.Tensor([2 * torch.pi]))) * \
               torch.exp(-1 / 2 * (((x - mu) / sigma) ** 2))

    def g1(x, sigma1=sigma1):
        return gaussian_pdf(x / sigma1)

    def g2(x, sigma2=sigma2):
        levy = LevyStable()
        return levy.pdf(x / sigma2, alpha, beta=0)

    def g3(x, sigma1=sigma1):
        return -x / sigma1 * gaussian_pdf(x / sigma1)

    y = 1 / sigma1 * cft(g3, g2, x) / cft(g1, g2, x)
    return y