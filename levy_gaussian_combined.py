import torch
from torch_dictionary import TorchDictionary
from torchquad import set_up_backend  # Necessary to enable GPU support
from torchquad import Simpson # The available integrators

if torch.cuda.is_available():
    set_up_backend("torch", data_type="float32")

class LevyGaussian:

    def __init__(self, alpha, sigma_1, sigma_2, beta=0, t0=30, Fs=50, type="cft"):
        self.alpha = alpha
        self.beta = beta

        if type == "cft":
            self.score_dict = self._get_score_dict_cft(alpha, sigma_1, sigma_2, beta, t0, Fs)
            self.score_dict_large_Fs = self._get_score_dict_fft(alpha, sigma_1, sigma_2, beta, t0, Fs=300)
        elif type == "fft":
            self.score_dict = self._get_score_dict_fft(alpha, sigma_1, sigma_2, beta, t0, Fs=300)
            self.score_dict_large_Fs = self.score_dict
        else:
            raise RuntimeError(f"type :{type} isn't implemented")


    def score(self, x: torch.Tensor):
        score = self.score_dict.get(x, linear_approx=True)
        score_for_large_x = self.score_dict_large_Fs.get(x, linear_approx=True)

        score[torch.abs(x) > 18] = score_for_large_x[torch.abs(x) > 18]
        return score

    def _get_score_dict_cft(self, alpha, sigma_1, sigma_2, beta, t0, Fs):
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

        return TorchDictionary(keys=f, values=score)

    def _get_score_dict_fft(self, alpha, sigma_1, sigma_2, beta, t0, Fs):

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

        return TorchDictionary(keys=f, values=score)