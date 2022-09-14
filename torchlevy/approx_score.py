import torch
from functools import lru_cache
from .levy import LevyStable


def get_approx_score(x, alpha):
    c, t = _get_c_t(alpha)
    ret = torch.zeros_like(x, dtype=torch.float32)
    ret[x >= 0] = (-c * (x ** t))[x >= 0]
    ret[x < 0] = (c * ((-x) ** t))[x < 0]
    return ret


@lru_cache()
def _get_c_t(alpha):
    def get_extreme_pts(func, x=torch.arange(-15, 15, 0.01)):
        y = func(x)
        dy = y[1:] - y[:-1]
        idx = ((dy[1:] * dy[:-1]) < 0).nonzero()
        return x[idx + 1]

    levy = LevyStable()
    func = lambda x: levy.score(x, alpha=alpha)
    pts = get_extreme_pts(func)

    x = torch.linspace(0, pts[1].item(), 100)
    t = torch.linspace(0.1, 1, 100).reshape(-1, 1)
    c = torch.linspace(0.1, 10, 100).reshape(-1, 1, 1)

    res = torch.abs(func(x) - (-c * (x ** t)))
    res = torch.sum(res, axis=2)

    t_idx = torch.argmin(res) % 100
    c_idx = torch.argmin(res) // 100

    return c.ravel()[c_idx], t.ravel()[t_idx]
