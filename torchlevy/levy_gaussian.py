import torch
from .torch_dictionary import TorchDictionary
from torchquad import set_up_backend  # Necessary to enable GPU support
from torchquad import Simpson  # The available integrators
from functools import lru_cache
from .util import gaussian_score

import matplotlib.pyplot as plt
import warnings
from typing import Union


class LevyGaussian:
    def __init__(self, alpha, sigma_1, sigma_2, beta=0, t0=30, Fs=100, type="fft"):
        self.alpha = alpha
        self.beta = beta

        if beta != 0:
            raise NotImplementedError(f"beta != 0 not yet implemented")

        if type == "cft":
            warnings.warn("cft will be deprecated. please use fft")

        if type in ["cft", "fft"]:
            self.score_dict = _get_score_dict_fft(alpha, sigma_1, sigma_2, beta, t0, Fs)
            self.score_dict_large_Fs = _get_score_dict_fft(alpha, sigma_1, sigma_2, beta, t0, Fs=300)
        else:
            raise RuntimeError(f"type :{type} isn't implemented")

    def score(self, x: torch.Tensor):

        if self.alpha == 2:
            return gaussian_score(x)

        score = self.score_dict.get(x, linear_approx=True)
        score_for_large_x = self.score_dict_large_Fs.get(x, linear_approx=True)

        score[torch.abs(x) > 18] = score_for_large_x[torch.abs(x) > 18]
        return score


def levy_gaussian_score(alpha: float, x: torch.Tensor, sigma1s: Union[list, torch.Tensor],
                        sigma2s: Union[list, torch.Tensor], beta=0, t0=30, Fs=100):
    score = torch.zeros_like(x)

    for i, (s1, s2) in enumerate(zip(sigma1s, sigma2s)):
        score_dict = _get_score_dict_fft(alpha, s1, s2, beta, t0, Fs)
        score[i] = score_dict.get(x[i], linear_approx=True)

    return score


@lru_cache(maxsize=1050)
def _get_score_dict_fft(alpha, sigma_1, sigma_2, beta, t0, Fs):
    def cft(g, f):
        """Numerically evaluate the Fourier Transform of g for the given frequencies"""

        simp = Simpson()
        intg = simp.integrate(lambda t: g(t) * torch.exp(-1j * f * t), dim=1, N=10001,
                              integration_domain=[[-100, 100]])
        return intg

    def g1(t):
        return torch.exp(-torch.pow(torch.abs(t), alpha))

    def g2(t):
        return -(1j * t) * torch.abs(t + 1e-20) ** (alpha - 2) * torch.exp(
            -1 / 2 * torch.pow(torch.abs(t * (sigma_1 / sigma_2)), 2)) * torch.exp(-torch.pow(torch.abs(t), alpha))

    t = torch.arange(-t0, t0, 1. / Fs)
    f = torch.linspace(-Fs / 2, Fs / 2, len(t))

    G_exact1 = cft(g1, f).real
    G_exact2 = cft(g2, f).real

    score = G_exact2 / G_exact1

    return TorchDictionary(keys=f, values=score)
