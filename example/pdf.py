import torch
from torchlevy import LevyStable
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

range_ = 15
x = torch.arange(-range_, range_, 0.1)

levy = LevyStable()
# alpha = 2.0
alphas = [1.2, 1.5, 1.8]

for alpha in alphas:
    pdf = levy.pdf(x, alpha).cpu().numpy()

    plt.plot(x.cpu(), pdf, lw=2, label=f"alpha={alpha}")
    plt.xlim((-range_, range_))
    plt.ylim((0, 0.4))
    plt.legend()

plt.show()