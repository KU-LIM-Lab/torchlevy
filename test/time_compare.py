import sys
sys.path.append("../")

from scipy.stats import levy_stable
import torch
from torchlevy import LevyStable, util
import time


def test_sampling_time():
    alpha = 1.5
    beta = 0
    size = 10000000

    levy = LevyStable()
    start = time.time()
    tmp2 = levy.sample(alpha, beta, size)
    print("\n")
    print("torch sampling takes ", time.time() - start, "s")

    start = time.time()
    tmp = levy_stable.rvs(alpha, beta, size=size)
    print("scipy sampling takes ", time.time() - start, "s")


def test_pdf_time():
    alpha = 1.5
    x = torch.arange(-10, 10, 0.01) # size = 2000

    levy = LevyStable()
    start = time.time()
    tmp = levy.pdf(x, alpha)
    print("\n")
    print(f"torch pdf evaluation takes {time.time() - start}s for {x.size()[0]} samples")

    x = x.cpu().detach()
    start = time.time()
    tmp = levy_stable.pdf(x, alpha, beta=0)
    print(f"scipy pdf evaluation takes {time.time() - start}s for {x.size()[0]} samples")


def test_score_time1():
    """ scipy vs torch """
    alpha = 1.5
    x = torch.arange(-10, 10, 0.01) # size = 2000

    levy = LevyStable()
    start = time.time()
    tmp = levy.score(x, alpha)
    print("\n")
    print(f"torch score evaluation takes {time.time() - start}s for {x.size()[0]} samples")

    x = x.cpu().detach()
    start = time.time()
    tmp = util.score_finite_diff(x, alpha)
    print(f"scipy score evaluation takes {time.time() - start}s for {x.size()[0]} samples")


def test_score_time2():
    """
    cft vs backpropagation
        cft             : fast and cache-applied
        backpropagation : slow and not cache-applied
    """
    alpha = 1.5
    x = torch.randn((3, 32, 32))
    # x = torch.randn((3, 128, 128))

    levy = LevyStable()
    start = time.time()
    tmp = levy.score(x, alpha)
    print("\n")
    print(f"first backpropagation score evaluation takes {time.time() - start}s")

    levy = LevyStable()
    start = time.time()
    tmp = levy.score(x, alpha)
    print(f"second backpropagation score evaluation takes {time.time() - start}s")

    levy = LevyStable()
    start = time.time()
    tmp = levy.score(x, alpha, type="cft")
    print(f"first cft score evaluation takes {time.time() - start}s")

    levy = LevyStable()
    start = time.time()
    tmp = levy.score(x, alpha, type="cft")
    print(f"second cft score evaluation takes {time.time() - start}s")


