import torch
from torchquad import Simpson  # The available integrators
from torchlevy import LevyStable

@torch.no_grad()
def levy_gaussian_score(x :torch.Tensor, alpha, sigma1, sigma2):
    if type(sigma1) not in [int, float]:
        sigma1 = sigma1.ravel()
    if type(sigma2) not in [int, float]:
        sigma2 = sigma2.ravel()
    def cft(f1, f2, z):
        """Numerically evaluate the Fourier Transform of g for the given frequencies"""
        simp = Simpson()
        score = simp.integrate(lambda y: f1(z - y) * f2(y), dim=1, N=333, integration_domain=[[-15, 15]])
        if torch.any(torch.abs(z) > 8):
            score_large_domain = simp.integrate(lambda y: f1(z - y) * f2(y), dim=1, N=999, integration_domain=[[-50, 50]])
            score[torch.abs(z) > 8] = score_large_domain[torch.abs(z) > 8]
        return score

    def gaussian_pdf(z, mu=0, sigma=1):
        return 1 / (sigma * torch.sqrt(torch.Tensor([2 * torch.pi]))) * \
               torch.exp(-1 / 2 * (((z - mu) / sigma) ** 2))

    def g1(z, sigma=sigma1):
        return gaussian_pdf(z / sigma)

    def g2(z, sigma=sigma2):
        levy = LevyStable()
        return levy.pdf(z / sigma, alpha, beta=0, is_cache=True)

    def g3(z, sigma=sigma1):
        return -z / sigma * gaussian_pdf(z / sigma)

    score = 1 / sigma1 * cft(g3, g2, x.ravel()) / cft(g1, g2, x.ravel())
    return score.reshape(x.size())