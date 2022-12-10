import torch
from Cython import inline
from torchquad import Simpson, MonteCarlo  # The available integrators
from torch.distributions.exponential import Exponential
from torchlevy import LevyGaussian
from .torch_dictionary import TorchDictionary
from .util import gaussian_score
from functools import lru_cache
from scipy.special import jv


class LevyStable:
    def pdf(self, x: torch.Tensor, alpha, beta=0, is_cache=False, is_isotropic=False):
        """
            calculate pdf through zolotarev thm
            ref. page 7, https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2894444_code545.pdf?abstractid=2894444&mirid=1
        """

        if is_isotropic:
            return self.pdf_isotropic(x, alpha)

        if is_cache:
            dense_dict, large_dict = _get_pdf_dict(alpha)
            ret = dense_dict.get(x)
            ret[abs(x) > 10] = large_dict.get(x)[abs(x) > 10]
            return ret

        x_flatten = x.reshape((-1))

        if alpha > 1 and beta == 0:
            ret = self._pdf_simple(x_flatten, alpha)
            return ret.reshape(x.shape)
        else:
            ret = self._pdf(x_flatten, alpha, beta)
            return ret.reshape(x.shape)

    def pdf_isotropic(self, x: torch.Tensor, alpha, beta=0):

        if beta != 0:
            raise NotImplementedError()

        norm_x = torch.norm(x, dim=1)
        def integrand(r):
            try:
                dim = x.shape[1]
            except:
                raise RuntimeError("dimension of x must >= 2")

            exponent = - r ** alpha + (dim / 2) * torch.log(r) \
                       - (dim/2 - 1) * torch.log(norm_x) - (dim/2) * torch.log(torch.tensor(2 * torch.pi))
            bessel_value = jv(dim / 2 - 1, (r * norm_x).cpu()).cuda()


            return torch.exp(exponent) * bessel_value

        simp = Simpson()
        ret = simp.integrate(integrand, dim=1, N=10000, integration_domain=[[1e-10, 10]])

        def integrand_x_around_0(r):
            """equation above results in nan when x ~= 0
            this is alternative integrand function when x -> 0"""
            try:
                dim = x.shape[1]
            except:
                raise RuntimeError("dimension of x must >= 2")

            exponent = -r ** alpha + (dim / 2) * torch.log(r) + (dim/2 - 1) * torch.log(r/2) - \
                       torch.lgamma(torch.tensor(dim / 2)) - (dim/2) * torch.log(torch.tensor(2 * torch.pi))

            return torch.exp(exponent)

        boundary = 1e-6
        if torch.any((x ** 2).sum(dim=1) < boundary):
            intg = simp.integrate(integrand_x_around_0, dim=1, N=100000, integration_domain=[[1e-10, 1000]])
            ret[(x ** 2).sum(dim=1) < boundary] = intg # when x ~= 0

        return ret

    def _pdf(self, x: torch.Tensor, alpha, beta):

        pi = torch.tensor(torch.pi, dtype=torch.float64)
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
                intg = simp.integrate(f, dim=1, N=10000, integration_domain=[[-xi + 1e-7, torch.pi / 2 - 1e-7]])

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

        ret = alpha * intg / torch.pi / torch.abs(torch.tensor(alpha - 1, dtype=torch.float64)) / x

        if torch.any(torch.abs(x) < 2e-2):
            gamma_func = lambda a: torch.exp(torch.special.gammaln(a))

            alpha = torch.tensor(alpha, dtype=torch.float64)

            ret[torch.abs(x) < 2e-2] = gamma_func(1 + 1 / alpha) / torch.pi

        return ret

    @torch.enable_grad()
    def score(self, x: torch.Tensor, alpha, beta=0, type="cft", is_isotropic=False, is_fdsm=True):

        if alpha == 2:
            return gaussian_score(x)

        if is_isotropic:
            # x.shape : (n, c, w, h)
            reshaped_x = x.reshape(x.shape[0], -1)  # dim2_x.shape : (n, c*w*h)
            norm_x = torch.norm(reshaped_x, dim=1)  # dim2_x.shape : (n,)
            dim = reshaped_x.shape[1]  # c*w*h

            def a(r_theta):
                r = r_theta[:, 0, None]
                theta = r_theta[:, 1, None]
                ret = -1j * torch.exp(-1j * r * norm_x * torch.cos(theta) - r ** alpha +
                                      (dim + alpha - 2) * torch.log(r) + (dim - 2) * torch.log(torch.sin(theta))) * \
                      torch.cos(theta)
                return ret.real

            def b(r_theta):
                r = r_theta[:, 0, None]
                theta = r_theta[:, 1, None]
                ret = torch.exp(-1j * r * norm_x * torch.cos(theta) - r ** alpha +
                                (dim - 1) * torch.log(r) + (dim - 2) * torch.log(torch.sin(theta)))
                return ret.real

            simp = Simpson()
            intg_a = simp.integrate(a, dim=2, N=300000,
                                    integration_domain=[[0, 10], [0, torch.pi]])  # shape : (n,)
            intg_b = simp.integrate(b, dim=2, N=300000,
                                    integration_domain=[[0, 10], [0, torch.pi]])  # shape : (n,)
            unit_x = reshaped_x / norm_x.reshape(-1, 1)  # (n, c*w*h)

            return (intg_a[:, None] / intg_b[:, None] * unit_x).reshape(x.shape)

        if is_fdsm and type != "cft":
            raise NotImplementedError("fdsm score is only implemented on cft")

        if type == "cft":
            levy_cft = LevyGaussian(alpha=alpha, sigma_1=0, sigma_2=1, beta=beta, is_fdsm=is_fdsm)
            return levy_cft.score(x)

        elif type == "default":
            def g1(t, x):
                return -torch.sin(t * x) * torch.exp(-torch.pow(t, alpha)) * t

            def g(t, x):
                return torch.cos(t * x) * torch.exp(-torch.pow(t, alpha))

            simp = Simpson()
            intg_g1 = simp.integrate(lambda t: g1(t, x.ravel()), dim=1, N=2501, integration_domain=[[0, 10]])
            intg_g = simp.integrate(lambda t: g(t, x.ravel()), dim=1, N=2501, integration_domain=[[0, 10]])

            return (intg_g1 / intg_g).reshape(x.shape)

        elif type == "backpropagation":

            if alpha > 1 and beta == 0:
                ret = self._score_simple(x.ravel(), alpha)
                ret = ret.reshape(x.shape)

                levy_fft = LevyGaussian(alpha=alpha, sigma_1=0, sigma_2=1, beta=beta, Fs=100)
                ret[torch.abs(x) > 18] = levy_fft.score(x)[torch.abs(x) > 18]

                return ret
            else:
                raise NotImplementedError("not yet implemented when alpha <= 1 or beta != 0 ")
        else:
            raise NotImplementedError(f"type : {type} not yet implemented")

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

    def sample(self, alpha, beta=0, size=1, loc=0, scale=1, type=torch.float32, reject_threshold: int = None,
               is_isotropic=False, clamp_threshold: int = None, clamp: int = None):

        assert 0 < alpha <= 2

        if isinstance(size, int):
            size_scalar = size
            size = (size,)
        else:
            size_scalar = 1
            for i in size:
                size_scalar *= i

        if is_isotropic and 0 < alpha < 2:
            assert not isinstance(size, int)

            num_sample = size[0]
            dim = int(size_scalar / num_sample)

            x = self._sample(alpha / 2, beta=1, size=num_sample * 2, type=type)
            x = x * 2 * torch.cos(torch.tensor([torch.pi * alpha / 4], dtype=torch.float64)) ** (2 / alpha)
            x = x.reshape(-1, 1)
            if clamp is not None:
                x = torch.clamp(x, -clamp, clamp)

            z = torch.randn(size=(num_sample * 2, dim))
            e = x ** (1 / 2) * z
            e = (e * scale) + loc
            if reject_threshold is not None:
                e = e[torch.norm(e, dim=1) < reject_threshold]
            if clamp_threshold is not None:
                indices = e.norm(dim=1) > clamp_threshold
                e[indices] = e[indices] / e[indices].norm(dim=1)[:, None] * clamp_threshold
            return e[:num_sample].reshape(size).to(type)

        else:
            e = self._sample(alpha, beta=beta, size=size_scalar * 2, type=type)
            e = (e * scale) + loc
            if reject_threshold is not None:
                e = e[torch.abs(e) < reject_threshold]
            return e[:size_scalar].reshape(size)

    def _sample(self, alpha, beta=0, size=1, type=torch.float32):

        def alpha1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
            return 2 / torch.pi * ((torch.pi / 2 + bTH) * tanTH
                                   - beta * torch.log((torch.pi / 2 * W * cosTH) / (torch.pi / 2 + bTH)))

        def beta0func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
            return (W / (cosTH / torch.tan(aTH) + torch.sin(TH)) *
                    ((torch.cos(aTH) + torch.sin(aTH) * tanTH) / W) ** (1 / alpha))

        def otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W):
            # alpha != 1 and beta != 0
            val0 = beta * torch.tan(torch.tensor([torch.pi * alpha / 2], dtype=torch.float64))
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
            return alpha1func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W).to(type)
        elif beta == 0:
            return beta0func(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W).to(type)
        else:
            return otherwise(alpha, beta, TH, aTH, bTH, cosTH, tanTH, W).to(type)


@lru_cache(maxsize=1050)
def _get_pdf_dict(alpha):
    levy = LevyStable()
    x = torch.arange(-10, 10, 0.01)
    pdf = levy.pdf(x, alpha)
    dense_dict = TorchDictionary(keys=x, values=pdf)

    x = torch.arange(-100, 100, 0.1)
    pdf = levy.pdf(x, alpha)
    large_dict = TorchDictionary(keys=x, values=pdf)

    return dense_dict, large_dict
