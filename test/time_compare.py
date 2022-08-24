
from scipy.stats import levy_stable
import numpy as np
import torch
from levy_stable_pytorch import LevyStable
from levy_gaussian_combined import LevyGaussian
import time
import util


def test_sampling_time():
    alpha = 1.5
    beta = 0
    size = 10000000

    levy = LevyStable()
    start = time.time()
    tmp2 = levy.sample(alpha, beta, size)
    print("torch sampling takes ", time.time() - start, "s")

    start = time.time()
    tmp = levy_stable.rvs(alpha, beta, size=size)
    print("scipy sampling takes ", time.time() - start, "s")
    print("\n")

def test_score_time():
    alpha = 1.5
    x = torch.arange(-10, 10, 0.01) # size = 2000

    levy = LevyStable()
    start = time.time()
    tmp = levy.score(x, alpha)
    print(f"torch score evaluation takes {time.time() - start}s for {x.size()[0]} samples")

    x = x.cpu().detach()
    start = time.time()
    tmp = util.score_finite_diff(x, alpha)
    print(f"scipy score evaluation takes {time.time() - start}s for {x.size()[0]} samples")
    print("\n")


def test_pdf_time():
    alpha = 1.5
    x = torch.arange(-10, 10, 0.01) # size = 2000

    levy = LevyStable()
    start = time.time()
    tmp = levy.pdf(x, alpha)
    print(f"torch pdf evaluation takes {time.time() - start}s for {x.size()[0]} samples")

    x = x.cpu().detach()
    start = time.time()
    tmp = levy_stable.pdf(x, alpha, beta=0)
    print(f"scipy pdf evaluation takes {time.time() - start}s for {x.size()[0]} samples")
    print("\n")

def test_levy_gaussian_score_time():
    alpha = 1.7
    x = torch.arange(-10, 10, 0.00001) # size = 2000000

    # continuous fourier transform
    start = time.time()
    levy_gaussian = LevyGaussian(alpha=alpha, sigma_1=1, sigma_2=1, type="cft")
    tmp = levy_gaussian.score(x)
    print("\n")
    print(f"cft score takes {time.time() - start}s for {x.size()[0]} samples")

    # fast fourier transform
    start = time.time()
    levy_gaussian = LevyGaussian(alpha=alpha, sigma_1=1, sigma_2=1, type="fft")
    tmp = levy_gaussian.score(x)
    print("\n")
    print(f"cft score takes {time.time() - start}s for {x.size()[0]} samples")


