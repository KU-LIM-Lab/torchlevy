import torch
from torchlevy import LevyStable
import matplotlib.pyplot as plt

def test_sample_clamp():
    for clamp in range(2, 100):
        levy = LevyStable()
        y2 = levy.sample(alpha=1.7, size=(100, 100, 100), clamp=clamp)
        assert(not torch.any(y2 > clamp))

test_sample_clamp()