import torch
from torchquad import Simpson, set_up_backend
from main import *
# Enable GPU support if available and set the floating point precision
set_up_backend("torch", data_type="float32")

for i in range(1):
    print(alpha_stable_pdf_zolotarev_simple(torch.tensor(float(i)), 1.7))