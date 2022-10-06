import torch
from functools import lru_cache
from .levy import LevyStable


def get_approx_score(x, alpha, is_mid_real_score=True):
    extreme_pts, c, t = _get_c_t(alpha)
    approx_score = torch.zeros_like(x, dtype=torch.float32)
    approx_score[x >= 0] = (-c * (x ** t))[x >= 0]
    approx_score[x < 0] = (c * ((-x) ** t))[x < 0]

    if is_mid_real_score and alpha < 2:
        levy = LevyStable()
        score = levy.score(x, alpha)
        approx_score[torch.abs(x) <= extreme_pts[-1] * 0.8] = score[[torch.abs(x) <= extreme_pts[-1] * 0.8]]

    return approx_score


@lru_cache()
def _get_c_t(alpha):
    def get_extreme_pts(func, x=torch.arange(-15, 15, 0.01)):
        y = func(x)
        dy = y[1:] - y[:-1]
        idx = ((dy[1:] * dy[:-1]) < 0).nonzero()
        return x[idx + 1]

    levy = LevyStable()
    func = lambda x: levy.score(x, alpha=alpha)
    extreme_pts = get_extreme_pts(func)

    x = torch.linspace(0, extreme_pts[-1].item(), 100)
    t = torch.linspace(0.1, 1, 1000).reshape(-1, 1)
    c = torch.linspace(0.1, 10, 1000).reshape(-1, 1, 1)

    res = torch.abs(func(x) - (-c * (x ** t)))
    res = torch.sum(res, axis=2)

    t_idx = torch.argmin(res) % 1000
    c_idx = torch.argmin(res) // 1000

    return extreme_pts, c.ravel()[c_idx], t.ravel()[t_idx]



def get_approx_score2(x, alpha, is_mid_real_score=True):
    t = alpha * 0.5
    extreme_pts, c, = _get_c(alpha, t)
    approx_score = torch.zeros_like(x, dtype=torch.float32)
    approx_score[x >= 0] = (-c * (x ** t))[x >= 0]
    approx_score[x < 0] = (c * ((-x) ** t))[x < 0]

    if is_mid_real_score and alpha < 2:
        levy = LevyStable()
        score = levy.score(x, alpha)
        approx_score[torch.abs(x) <= extreme_pts[-1] * 0.8] = score[[torch.abs(x) <= extreme_pts[-1] * 0.8]]

    return approx_score


def get_approx_score3(x, alpha, is_mid_real_score=True):
    t = alpha * 0.5 - 0.1
    extreme_pts, c, = _get_c(alpha, t)
    approx_score = torch.zeros_like(x, dtype=torch.float32)
    approx_score[x >= 0] = (-c * (x ** t))[x >= 0]
    approx_score[x < 0] = (c * ((-x) ** t))[x < 0]

    if is_mid_real_score and alpha < 2:
        levy = LevyStable()
        score = levy.score(x, alpha)
        approx_score[torch.abs(x) <= extreme_pts[-1] * 0.8] = score[[torch.abs(x) <= extreme_pts[-1] * 0.8]]

    return approx_score



@lru_cache()
def _get_c(alpha, t):
    def get_extreme_pts(func, x=torch.arange(-15, 15, 0.01)):
        y = func(x)
        dy = y[1:] - y[:-1]
        idx = ((dy[1:] * dy[:-1]) < 0).nonzero()
        return x[idx + 1]

    levy = LevyStable()
    func = lambda x: levy.score(x, alpha=alpha)
    extreme_pts = get_extreme_pts(func)

    x = torch.linspace(0, extreme_pts[1].item(), 100)
    c = torch.linspace(0.01, 10, 1000).reshape(-1, 1)

    res = torch.abs(func(x) - (-c * (x ** t)))
    res = torch.sum(res, axis=-1)

    c_idx = torch.argmin(res)

    return extreme_pts, c.ravel()[c_idx]

