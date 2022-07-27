
import torch
import numpy as np
from Cython import inline
from torchquad import set_up_backend  # Necessary to enable GPU support
from torchquad import Simpson # The available integrators
from torch.distributions.exponential import Exponential

if torch.cuda.is_available():
    set_up_backend("torch", data_type="float32")

class LevyStable:

    def pdf(self, x: torch.Tensor, alpha, beta=0):
        """
            calculate pdf through zolotarev thm
            ref. page 7, https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2894444_code545.pdf?abstractid=2894444&mirid=1
        """

        if alpha > 1 and beta==0:
            return self._pdf_simple(x, alpha)


        pi = torch.tensor(torch.pi)
        zeta = -beta * torch.tan(pi * alpha / 2.)

        if alpha != 1:
            x0 = x + zeta  # convert to S_0 parameterization
            xi = torch.arctan(-zeta) / alpha

            if x0 == zeta:
                gamma_func = lambda a: torch.exp(torch.special.gammaln(a))
                alpha = torch.tensor(alpha)
                alpha.requires_grad_()

                return gamma_func(1 + 1 / alpha) * torch.cos(xi) / torch.pi / ((1 + zeta ** 2) ** (1 / alpha / 2))

            elif x0 > zeta:

                def V(theta):
                    return torch.cos(alpha * xi) ** (1 / (alpha - 1)) * \
                           (torch.cos(theta) / torch.sin(alpha * (xi + theta))) ** (alpha / (alpha - 1)) * \
                           (torch.cos(alpha * xi + (alpha - 1) * theta) / torch.cos(theta))

                @inline
                def g(theta):
                    return V(theta) * (x0 - zeta) ** (alpha / (alpha - 1))

                @inline
                def f(theta):
                    g_ret = g(theta)
                    g_ret = torch.nan_to_num(g_ret, posinf=0, neginf=0)

                    return g_ret * torch.exp(-g_ret)

                # spare calculating integral on null set
                # use isclose as macos has fp differences
                if torch.isclose(-xi, pi / 2, rtol=1e-014, atol=1e-014):
                    return 0.

                simp = Simpson()
                intg = simp.integrate(f, dim=1, N=101, integration_domain=[[-xi + 1e-7, torch.pi / 2 - 1e-7]])

                return alpha * intg / torch.pi / torch.abs(torch.tensor(alpha - 1)) / (x0 - zeta)

            else:
                return self.pdf(-x, alpha, -beta)
        else:
            raise NotImplementedError("This function doesn't handle when alpha==1")

    def _pdf_simple(self, x: torch.Tensor, alpha):
        """
            simplified version of func `pdf_zolotarev`,
            assume alpha > 1 and beta = 0
        """
        assert (alpha > 1)

        x = torch.abs(x)

        def V(theta):
            return (torch.cos(theta) / torch.sin(alpha * theta)) ** (alpha / (alpha - 1)) * \
                   (torch.cos((alpha - 1) * theta) / torch.cos(theta))

        def g(theta):
            return V(theta) * x ** (alpha / (alpha - 1))

        def f(theta):
            g_ret = g(theta)
            g_ret = torch.nan_to_num(g_ret, posinf=0, neginf=0)

            return g_ret * torch.exp(-g_ret)

        simp = Simpson()
        intg = simp.integrate(f, dim=1, N=999, integration_domain=[[1e-7, torch.pi / 2 - 1e-7]])

        ret = alpha * intg / np.pi / torch.abs(torch.tensor(alpha - 1)) / x

        if torch.any(x == 0):
            gamma_func = lambda a: torch.exp(torch.special.gammaln(a))

            alpha = torch.tensor(alpha)

            ret[x == 0] = gamma_func(1 + 1 / alpha) / torch.pi

        return ret

    def score(self, x: torch.Tensor, alpha, beta=0):
        if alpha > 1 and beta==0:
            return self._score_simple(x, alpha)
        else:
            raise NotImplementedError("not yet implemented when alpha <= 1 or beta != 0 ")

    def _score_simple(self, x: torch.Tensor, alpha):

        x.requires_grad_()
        log_likelihood = torch.log(self._pdf_simple(x, alpha))
        grad = torch.autograd.grad(log_likelihood.sum(), x, allow_unused=True)[0]

        # code above doesn't well estimate the score when |x| < 0.05
        # so, do linear approximation when |x| < 0.05
        if torch.any(torch.abs(x) < 0.05):
            grad_0_05 = self._score_simple(torch.tensor(0.05), alpha)
            grad[torch.abs(x) < 0.05] = x[torch.abs(x) < 0.05] * 20 * grad_0_05

        return grad

    def sample(self, alpha, beta=0, size=None):
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



