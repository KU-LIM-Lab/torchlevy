import sys
sys.path.append("../")

from torchlevy.approx_score import get_approx_score, get_extreme_pts
from torchlevy import LevyStable, levy_stable
import torch
import matplotlib.pyplot as plt

def test_approx_score():
    alphas = torch.arange(1, 2.001, 0.001)
    x = torch.arange(-5, 5, 0.1)

    for alpha in alphas:
        a = get_approx_score(x, alpha)

def plot_score(alpha=1.9930000305175781):

    
    x = torch.arange(-15, 15, 0.01)

    levy_score_default = levy_stable.score(x, alpha=alpha, type="default").detach().cpu().numpy()
    plt.plot(x.cpu().numpy(), levy_score_default, 'c-', lw=2, label="default")
    plt.show()

def test_get_extreme_pts(alpha=1.590000033378601):

    
    func = lambda x: levy_stable.score(x, alpha=alpha)
    extreme_pts = get_extreme_pts(func)

    if len(extreme_pts) != 2:
        raise RuntimeError(f"extreme_pts: {extreme_pts}")

if __name__ == "__main__":
    # plot_score()
    test_approx_score()


