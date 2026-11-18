"""Task 4: compare my engine vs PyTorch autograd vs numerical gradients."""
import numpy as np
import torch
from engine import Value

rng = np.random.default_rng(0)
a0, b0, c0 = (rng.normal(size=6) for _ in range(3))

# --- my engine: scalar Values, one per element, then sum ---
A = [Value(x) for x in a0]; B = [Value(x) for x in b0]; C = [Value(x) for x in c0]
out = sum(((a * b + c) ** 2).relu() for a, b, c in zip(A, B, C))
out.backward()
mine = np.array([[v.grad for v in L] for L in (A, B, C)])

# --- PyTorch autograd (float64 so precision is comparable) ---
ta, tb, tc = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (a0, b0, c0))
((ta * tb + tc) ** 2).relu().sum().backward()
tor = np.array([t.grad.numpy() for t in (ta, tb, tc)])

# --- numerical gradients: central differences ---
def f(a, b, c):
    return np.sum(np.maximum((a * b + c) ** 2, 0))

def numgrad(i, h=1e-5):
    args = [a0.copy(), b0.copy(), c0.copy()]
    g = np.zeros_like(a0)
    for k in range(len(a0)):
        args[i][k] += h; hi = f(*args)
        args[i][k] -= 2 * h; lo = f(*args)
        args[i][k] += h
        g[k] = (hi - lo) / (2 * h)
    return g
num = np.array([numgrad(i) for i in range(3)])

print("value  mine:", out.data, " torch:", ((ta * tb + tc) ** 2).relu().sum().item())
print("max |mine - torch|    =", np.abs(mine - tor).max())
print("max |mine - numerical| =", np.abs(mine - num).max())
assert np.abs(mine - tor).max() < 1e-6 and np.abs(mine - num).max() < 1e-6
print("PASS: both differences < 1e-6")