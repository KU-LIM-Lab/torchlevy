import torch
from torchlevy import LevyStable, stable_dist
import matplotlib.pyplot as plt

range_ = 4
x = torch.arange(-range_, range_ + 1, 0.1)



alphas = [1.5]
plt.figure(figsize=(10, 3))


for alpha in alphas:
    pdf = stable_dist.pdf(x, alpha).cpu().numpy()

    plt.plot(x.cpu(), pdf, lw=5, label=f"alpha={alpha}", color='k')
    plt.xlim((-range_, range_))
    plt.ylim((0, 0.4))
    # plt.legend()

ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.ylim(0, 0.3)
plt.show()