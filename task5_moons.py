"""Task 5: train 2 -> 8 -> 1 (ReLU hidden, sigmoid out, BCE) on make_moons: my engine vs PyTorch."""
import time, statistics
import numpy as np
import torch
from sklearn.datasets import make_moons
from engine import Value

X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
EPOCHS, LR, RUNS, EPS = 100, 0.5, 5, 1e-9

rng = np.random.default_rng(0)       # identical starting weights for both implementations
W1 = rng.normal(0, 0.7, (8, 2)); b1 = np.zeros(8)
W2 = rng.normal(0, 0.7, (1, 8)); b2 = np.zeros(1)


def train_mine():
    w1 = [[Value(v) for v in row] for row in W1]; bb1 = [Value(v) for v in b1]
    w2 = [Value(v) for v in W2[0]];               bb2 = Value(b2[0])
    params = [p for r in w1 for p in r] + bb1 + w2 + [bb2]
    for _ in range(EPOCHS):
        losses, correct = [], 0
        for (x0, x1), t in zip(X, y):
            h = [(r[0] * x0 + r[1] * x1 + b).relu() for r, b in zip(w1, bb1)]
            p = (sum(w * hh for w, hh in zip(w2, h)) + bb2).sigmoid()
            losses.append(-(t * (p + EPS).log() + (1 - t) * (1 - p + EPS).log()))
            correct += (p.data > 0.5) == t
        loss = sum(losses) * (1 / len(X))
        for q in params: q.grad = 0.0
        loss.backward()
        for q in params: q.data -= LR * q.grad
    return loss.data, correct / len(X)


def train_torch():
    m = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1), torch.nn.Sigmoid())
    with torch.no_grad():
        m[0].weight.copy_(torch.tensor(W1)); m[0].bias.copy_(torch.tensor(b1))
        m[2].weight.copy_(torch.tensor(W2)); m[2].bias.copy_(torch.tensor(b2))
    m = m.double()
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float64).unsqueeze(1)
    opt = torch.optim.SGD(m.parameters(), lr=LR); lossf = torch.nn.BCELoss()
    for _ in range(EPOCHS):
        opt.zero_grad(); p = m(Xt); loss = lossf(p, yt); loss.backward(); opt.step()
    return loss.item(), ((p > 0.5).double() == yt).double().mean().item()


for name, fn in (("my engine", train_mine), ("PyTorch", train_torch)):
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter(); res = fn(); times.append(time.perf_counter() - t0)
    print(f"{name:10s} median {statistics.median(times):8.4f} s over {RUNS} runs | final loss {res[0]:.4f} acc {res[1]:.3f}")