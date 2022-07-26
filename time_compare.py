
from scipy.stats import levy_stable
import numpy as np
import torch
from levy_stable_pytorch import LevyStable
import time
import util


def sampling_time_compare():
    alpha = 1.5
    beta = 0
    size = 10000000

    start = time.time()
    tmp = levy_stable.rvs(alpha, beta, size=size)
    print("scipy sampling takes ", time.time() - start, "s")

    levy = LevyStable()
    start = time.time()
    tmp2 = levy.sample(alpha, beta, size)
    print("torch sampling takes ", time.time() - start, "s")
    print("\n")

def score_time_compare():
    alpha = 1.5
    x = torch.arange(-10, 10, 0.01) # size = 2000
    start = time.time()

    tmp = util.score_finite_diff(x, alpha)
    print(f"scipy score evaluation takes {time.time() - start}s for {x.size()[0]} samples")

    levy = LevyStable()
    x = torch.arange(-10, 10, 0.0001) # size = 2000000
    start = time.time()
    tmp = levy.score_simple(x, alpha)
    print(f"torch score evaluation takes {time.time() - start}s for {x.size()[0]} samples")
    print("\n")


def pdf_time_compare():
    alpha = 1.5
    x = torch.arange(-10, 10, 0.01) # size = 2000
    start = time.time()
    tmp = levy_stable.pdf(x, alpha, beta=0)
    print(f"scipy pdf evaluation takes {time.time() - start}s for {x.size()[0]} samples")

    levy = LevyStable()
    x = torch.arange(-10, 10, 0.0001) # size = 2000000
    start = time.time()
    tmp = levy.pdf_simple(x, alpha)
    print(f"torch pdf evaluation takes {time.time() - start}s for {x.size()[0]} samples")
    print("\n")


# sampling_time_compare()
# score_time_compare()
# pdf_time_compare()