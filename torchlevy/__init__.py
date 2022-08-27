from .levy_gaussian import LevyGaussian
from .levy import LevyStable

from torchquad import set_up_backend  # Necessary to enable GPU support
import torch

if torch.cuda.is_available():
    set_up_backend("torch", data_type="float32")