import torch
from levy_stable_pytorch import LevyStable

levy = LevyStable()

alpha = 1.7
x = torch.randn(size=(3, 200, 200))
size = (3, 200, 200)

# x10 faster version of levy_stable.rvs()
sample = levy.sample(alpha, 0, size)

# score function
score = levy.score(x, alpha)

# likelihood
likelihood = levy.pdf(x, alpha)


import torch
from levy_gaussian_combined import LevyGaussian

alpha = 1.7
x = torch.randn(size=(3, 200, 200))

sigma_1 = 1
sigma_2 = 1

# gaussian + levy score via "continuous" fourier transform
# more accurate, but slower
levy_gaussian = LevyGaussian(alpha, sigma_1, sigma_2, type="cft")
score = levy_gaussian.score(x)

# gaussian + levy score via "fast" fourier transform
# less accurate, but faster
levy_gaussian = LevyGaussian(alpha, sigma_1, sigma_2, type="fft")
score = levy_gaussian.score(x)