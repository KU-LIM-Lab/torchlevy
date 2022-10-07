import torch
from functools import lru_cache
from .levy import LevyStable
import numpy as np


def get_approx_score(x, alpha, is_mid_real_score=True):
    if alpha == 2:
        return - x / 2

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

    levy = LevyStable()
    func = lambda x: levy.score(x, alpha=alpha)
    extreme_pts = get_extreme_pts(func)

    if len(extreme_pts) != 2:
        idx = len(extreme_pts) // 2
        extreme_pts = [-extreme_pts[idx], extreme_pts[idx]]

    x = torch.linspace(0, extreme_pts[-1].item(), 100)
    t = torch.linspace(0.1, 1, 1000).reshape(-1, 1)
    c = torch.linspace(0.1, 10, 1000).reshape(-1, 1, 1)

    res = torch.abs(func(x) - (-c * (x ** t)))
    res = torch.sum(res, axis=2)

    t_idx = torch.argmin(res) % 1000
    c_idx = torch.argmin(res) // 1000

    return extreme_pts, c.ravel()[c_idx], t.ravel()[t_idx]



def get_approx_score2(x, alpha, is_mid_real_score=True):
    if alpha == 2:
        return - x / 2

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
    if alpha == 2:
        return - x / 2

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

    levy = LevyStable()
    func = lambda x: levy.score(x, alpha=alpha)
    extreme_pts = get_extreme_pts(func)

    x = torch.linspace(0, extreme_pts[1].item(), 100)
    c = torch.linspace(0.01, 10, 1000).reshape(-1, 1)

    res = torch.abs(func(x) - (-c * (x ** t)))
    res = torch.sum(res, axis=-1)

    c_idx = torch.argmin(res)

    return extreme_pts, c.ravel()[c_idx]


def get_extreme_pts(func, x=torch.arange(-10, 10, 0.01)):
    y = func(x)
    dy = y[1:] - y[:-1]
    indice = ((dy[1:] * dy[:-1]) <= 0).nonzero()

    # print(1111, indice)
    # remove duplicate
    new_indice = []
    for i in range(len(indice)-1):
        if indice[i] + 1 == indice[i+1]:
            continue
        else:
            new_indice.append(indice[i])

    new_indice.append(indice[-1])
    new_indice = torch.Tensor(new_indice).long()
    # print(2222, new_indice)

    return x[new_indice + 1]