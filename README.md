# Levy-Stable-Pytorch
This repository provides torch-based pdf, score calculation and sampling of **[levy stable distribution](https://en.wikipedia.org/wiki/L%C3%A9vy_distribution)**.
## Setup


```python
pip install -r requirements.txt
```

## Simple Tutorial Code 



```python
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
```


## Performance
* Result from [time_compare.py](https://github.com/UNIST-LIM-Lab/levy-stable-pytorch/blob/master/time_compare.py)

### batman server

|  | sampling | score | likelihood |
| --- | --- | --- | --- |
| scipy | 9.055s | 58.837s | 14.714s |
| levy-stable-pytorch | 0.009s (x1000 faster) | 0.026s (x2000 faster) | 0.003s (x4000 faster) |

### m1 mac mini

|  | sampling | score | likelihood |
| --- | --- | --- | --- |
| scipy | 1.843s | 12.668s | 3.188s |
| levy-stable-pytorch | 0.370s (x5 faster) | 0.029s (x400 faster) | 0.008s (x400 faster) |


## Gaussian + Levy Tutorial Code
```python
import torch
from levy_gaussian_combined import LevyGaussian

alpha = 1.7
x = torch.randn(size=(3, 200, 200))

levy_gaussian = LevyGaussian()

# gaussian + levy score via continuous fourier transform
score = levy_gaussian.score_cft(x, alpha, sigma_1=1, sigma_2=1)

# gaussian + levy score via fast fourier transform
score = levy_gaussian.score_fft(x, alpha, sigma_1=1, sigma_2=1)

```

