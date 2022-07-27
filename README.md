# Levy-Stable-Pytorch
This repository provides torch-based pdf, score calculation and sampling of **[levy stable distribution](https://en.wikipedia.org/wiki/L%C3%A9vy_distribution)**.
## Setup


```python
pip install -r requirements.txt
```

## Simple Tutorial Code



```python
from levy_stable_pytorch import LevyStable

levy = LevyStable()

x = torch.arange(-10, 10, 0.0001) # size = 2000000
size = 10000

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