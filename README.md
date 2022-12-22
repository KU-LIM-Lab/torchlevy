# TorchLevy
TorchLevy is python package that provides torch-based pdf, score calculation and sampling of **[symmetric alpha stable distribution](https://en.wikipedia.org/wiki/Stable_distribution)**.
## Setup


```python
pip install git+https://github.com/UNIST-LIM-Lab/torchlevy.git 
```

## Simple Tutorial Code

```python
import torch
from torchlevy import LevyStable

levy = LevyStable()

alpha=1.7
x = torch.randn(size=(3, 200, 200))

# likelihood
likelihood = levy.pdf(x, alpha)

# score function
score = levy.score(x, alpha)

# x10 faster version of levy_stable.rvs()
sample = levy.sample(alpha, beta=0, size=(3, 200, 200))

```


## Performance
* Result from [time_compare.py](https://github.com/UNIST-LIM-Lab/torchlevy/blob/master/time_compare.py)

### RTX 3090 + AMD EPYC 7702

|  | sampling | score | likelihood |
| --- | --- | --- | --- |
| scipy | 9.055s | 58.837s | 14.714s |
| torchlevy | 0.009s (x1000 faster) | 0.026s (x2000 faster) | 0.003s (x4000 faster) |

### mac mini M1 

|  | sampling | score | likelihood |
| --- | --- | --- | --- |
| scipy | 1.843s | 12.668s | 3.188s |
| torchlevy | 0.370s (x5 faster) | 0.029s (x400 faster) | 0.008s (x400 faster) |


## Gaussian + Levy Tutorial Code
```python
import torch
from torchlevy import LevyGaussian

x = torch.randn(size=(3, 200, 200))

# score of gaussian+levy via fourier transform
levy_gaussian = LevyGaussian(alpha=1.7, sigma_1=1, sigma_2=1)
score = levy_gaussian.score(x)


```
```python
import torch
from torchlevy import levy_gaussian_score

alpha = 1.8
x = torch.linspace(-10, 10, 100)
sigma1 = torch.ones(x.size()) * 0.5
sigma2 = torch.ones(x.size()) * 0.5

y = levy_gaussian_score(x, alpha=alpha, sigma1=sigma1, sigma2=sigma2)
```

