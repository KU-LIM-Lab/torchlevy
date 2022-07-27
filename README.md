# Levy-Stable-Pytorch

## setup


```python
pip install -r requirements.txt
```

## simple tutorial code



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

- 더 자세한 코드는 [time_compare.py](https://github.com/UNIST-LIM-Lab/levy-distribution-pytorch/blob/master/time_compare.py) 를 참고해 주세요.