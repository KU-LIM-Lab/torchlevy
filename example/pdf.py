import torch
from torchlevy import LevyStable, stable_dist
import matplotlib.pyplot as plt

range_ = 15
x = torch.arange(-range_, range_, 0.1)


alphas = [1.2, 1.5, 1.8]

for alpha in alphas:
    pdf = stable_dist.pdf(x, alpha).cpu().numpy()

    plt.plot(x.cpu(), pdf, lw=2, label=f"alpha={alpha}")
    plt.xlim((-range_, range_))
    plt.ylim((0, 0.4))
    plt.legend()

plt.show()