import sys
sys.path.append("../")

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
alpha = 1.7

pdf = levy.pdf(x, alpha).cpu().numpy()
sample = levy.sample(alpha, size=10000, clamp=3).cpu().numpy()
score = levy.score(x, alpha).detach().cpu().numpy()

plt.figure(figsize=(12, 3))

plt.subplot(131)
plt.plot(x.cpu(), pdf, 'r-', lw=2, label="pdf")
plt.xlim((-range_, range_))
plt.ylim((0, 0.3))
plt.legend()

plt.subplot(132)
plt.subplots_adjust(left=0.15)
plt.hist(sample, 2000, facecolor='blue', alpha=0.5)
plt.xlim((-range_, range_))
plt.legend()

plt.subplot(133)
plt.plot(x.cpu(), score, 'g-', lw=2, label="score")
plt.xlim((-range_, range_))
plt.legend()

plt.show()