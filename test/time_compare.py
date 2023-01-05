import sys
sys.path.append("../")

from scipy.stats import levy_stable
import torch
from torchlevy import LevyStable, stable_dist, util
import time


def test_sampling_time():
    alpha = 1.5
    beta = 0
    size = 10000000

    
    start = time.time()
    tmp2 = stable_dist.sample(alpha, beta, size)
    print("\n")
    print("torch sampling takes ", time.time() - start, "s")

    start = time.time()
    tmp = levy_stable.rvs(alpha, beta, size=size)
    print("scipy sampling takes ", time.time() - start, "s")


def test_pdf_time():
    alpha = 1.5
    x = torch.arange(-10, 10, 0.01) # size = 2000

    
    start = time.time()
    tmp = stable_dist.pdf(x, alpha)
    print("\n")
    print(f"torch pdf evaluation takes {time.time() - start}s for {x.size()[0]} samples")

    x = x.cpu().detach()
    start = time.time()
    tmp = stable_dist.pdf(x, alpha, beta=0)
    print(f"scipy pdf evaluation takes {time.time() - start}s for {x.size()[0]} samples")


def test_score_time1():
    """ scipy vs torch """
    alpha = 1.5
    x = torch.arange(-10, 10, 0.01) # size = 2000

    
    start = time.time()
    tmp = stable_dist.score(x, alpha)
    print("\n")
    print(f"torch score evaluation takes {time.time() - start}s for {x.size()[0]} samples")

    x = x.cpu().detach()
    start = time.time()
    tmp = util.score_finite_diff(x, alpha)
    print(f"scipy score evaluation takes {time.time() - start}s for {x.size()[0]} samples")


def test_score_time2():
    """
    default vs cft vs backpropagation
    """
    alpha = 1.5
    x = torch.randn((3, 32, 32))
    # x = torch.randn((3, 128, 128))

    
    start = time.time()
    tmp = stable_dist.score(x, alpha, type="backpropagation")
    print("\n")
    print(f"first backpropagation score evaluation takes {time.time() - start}s")

    
    start = time.time()
    tmp = stable_dist.score(x, alpha, type="backpropagation")
    print(f"second backpropagation score evaluation takes {time.time() - start}s")

    
    start = time.time()
    tmp = stable_dist.score(x, alpha, type="default")
    print("")
    print(f"first default score evaluation takes {time.time() - start}s")

    
    start = time.time()
    tmp = stable_dist.score(x, alpha, type="default")
    print(f"second default score evaluation takes {time.time() - start}s")

    
    start = time.time()
    tmp = stable_dist.score(x, alpha, type="cft")
    print("")
    print(f"first cft score evaluation takes {time.time() - start}s")

    
    start = time.time()
    tmp = stable_dist.score(x, alpha, type="cft")
    print(f"second cft score evaluation takes {time.time() - start}s")


if __name__ == "__main__":
    test_score_time2()

