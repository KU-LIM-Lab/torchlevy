import torch
from torch_dictionary import TorchDictionary
from torchquad import set_up_backend  # Necessary to enable GPU support
from torchquad import Simpson # The available integrators

if torch.cuda.is_available():
    set_up_backend("torch", data_type="float32")

class LevyGaussian:
    def score_cft(self, x: torch.Tensor, alpha, beta=0, t0=30, Fs=30, sigma_1=0, sigma_2=1):
        def cft(g, f):
            """Numerically evaluate the Fourier Transform of g for the given frequencies"""

            simp = Simpson()
            intg = simp.integrate(lambda t: g(t) * torch.exp(-2j * torch.pi * f * t), dim=1, N=10001,
                                  integration_domain=[[-5, 5]])
            return intg

        def g1(t):
            return torch.exp(-1 / 2 * (2 * torch.pi * t * sigma_1) ** 2) * torch.exp(
                -torch.pow(2 * torch.pi * torch.abs(t * sigma_2), alpha))

        def g2(t):
            return (-2j * torch.pi * t) * torch.exp(-1 / 2 * (2 * torch.pi * t * sigma_1) ** 2) * torch.exp(
                -torch.pow(2 * torch.pi * torch.abs(t * sigma_2), alpha))

        t = torch.arange(-t0, t0, 1. / Fs)
        f = torch.linspace(-Fs / 2, Fs / 2, len(t))

        G_exact1 = cft(g1, f)
        G_exact2 = cft(g2, f)

        score = G_exact2 / G_exact1

        score_dict = TorchDictionary(keys=f, values=score)
        return score_dict.get(x, linear_approx=True)

    def score_fft(self, x: torch.Tensor, alpha, beta=0, t0=10, Fs=30, sigma_1=0, sigma_2=1):
        def g1(t):
            return torch.exp(-1 / 2 * (2 * torch.pi * t * sigma_1) ** 2) * torch.exp(
                -torch.pow(2 * torch.pi * torch.abs(t * sigma_2), alpha))

        def g2(t):
            return (-2j * torch.pi * t) * torch.exp(-1 / 2 * (2 * torch.pi * t * sigma_1) ** 2) * torch.exp(
                -torch.pow(2 * torch.pi * torch.abs(t * sigma_2), alpha))

        t = torch.arange(-t0, t0, 1. / Fs)
        f = torch.linspace(-Fs / 2, Fs / 2, len(t))

        approx1 = torch.fft.fftshift(torch.fft.fft(g1(t)) * torch.exp(2j * torch.pi * f * t0) * 1 / Fs)
        approx2 = torch.fft.fftshift(torch.fft.fft(g2(t)) * torch.exp(2j * torch.pi * f * t0) * 1 / Fs)

        score = approx2 / approx1

        score_dict = TorchDictionary(keys=f, values=score)
        return score_dict.get(x, linear_approx=True)
