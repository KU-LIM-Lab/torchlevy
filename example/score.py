import torch
from torchlevy import LevyStable, levy_stable
import matplotlib.pyplot as plt

range_ = 5
x = torch.arange(-range_, range_, 0.1)


alphas = [1.2, 1.5, 1.8]

for alpha in alphas:
    score = levy_stable.score(x, alpha, is_fdsm=False).cpu()

    plt.plot(x.cpu(), score, lw=2, label=f"alpha={alpha}")
    plt.xlim((-range_, range_))
    plt.legend()

plt.show()