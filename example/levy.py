torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

range_ = 15
x = torch.arange(-range_, range_, 0.1)

levy = LevyStable()
# alpha = 2.0
alpha = 1.7
import torch
from levy_stable_pytorch.levy import LevyStable
import matplotlib.pyplot as plt

pdf = levy.pdf(x, alpha)
sample = levy.sample(alpha, size=10000).numpy()
score = levy.score(x, alpha).detach().numpy()

plt.figure(figsize=(12, 3))

plt.subplot(131)
plt.plot(x, pdf, 'r-', lw=2, label="pdf")
plt.xlim((-range_, range_))
plt.ylim((0, 0.3))
plt.legend()

plt.subplot(132)
plt.subplots_adjust(left=0.15)
plt.hist(sample, 2000, facecolor='blue', alpha=0.5)
plt.xlim((-range_, range_))
plt.legend()

plt.subplot(133)
plt.plot(x, score, 'g-', lw=2, label="score")
plt.xlim((-range_, range_))
plt.legend()

plt.show()