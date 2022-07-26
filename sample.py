
import torch
from torch.distributions.exponential import Exponential

def levy_sample(alpha, beta=0, size=None):
    def alpha1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        return 2 / torch.pi * ((torch.pi / 2 + bTH) * tanTH
                            - beta * torch.log((torch.pi / 2 * W * cosTH) / (torch.pi / 2 + bTH)))

    def beta0func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        return (W / (cosTH / torch.tan(aTH) + torch.sin(TH)) *
                ((torch.cos(aTH) + torch.sin(aTH) * tanTH) / W) ** (1 / alpha))

    def otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
        # alpha is not 1 and beta is not 0
        val0 = beta * torch.tan(torch.pi * alpha / 2)
        th0 = torch.arctan(val0) / alpha
        val3 = W / (cosTH / torch.tan(alpha * (th0 + TH)) + torch.sin(TH))
        res3 = val3 * ((torch.cos(aTH) + torch.sin(aTH) * tanTH -
                        val0 * (torch.sin(aTH) - torch.cos(aTH) * tanTH)) / W) ** (1 / alpha)
        return res3

    TH = torch.rand(size, dtype=torch.float64) * torch.pi - (torch.pi / 2.0)
    W = Exponential(torch.tensor([1.0])).sample([size]).reshape(-1)
    aTH = alpha * TH
    bTH = beta * TH
    cosTH = torch.cos(TH)
    tanTH = torch.tan(TH)

    if alpha == 1:
        return alpha1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W)
    elif beta == 0:
        return beta0func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W)
    else:
        return otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W)

