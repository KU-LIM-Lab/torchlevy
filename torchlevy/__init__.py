from .levy_gaussian import LevyGaussian
from .levy import LevyStable
from .levy_gaussian_score import levy_gaussian_score

from torchquad import set_up_backend  # Necessary to enable GPU support
import torch

if torch.cuda.is_available():
    set_up_backend("torch")