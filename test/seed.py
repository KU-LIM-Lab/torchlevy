import sys
sys.path.append("../")

import torch
from torchlevy import LevyStable, stable_dist

def test_seed_same():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    
    x = stable_dist.sample(alpha=1.7, size=10)
    y = torch.tensor([ 0.4381,  1.3218,  0.9782, -2.4093,  0.3261,  0.8505,  0.2408,  0.3322, 2.7384,  0.8296])
    assert(torch.all(torch.abs(x - y) < 1e-3))


if __name__ == "__main__":
    test_seed_same()
