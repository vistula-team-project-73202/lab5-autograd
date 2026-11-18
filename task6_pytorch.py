"""Task 6: inspect graph, eager vs compiled/traced, contiguity."""
import time, statistics
import torch
import torch.nn as nn

torch.manual_seed(0)


def bench(fn, warmup=10, iters=50, repeats=5):
    """median seconds per call, after warm-up"""
    for _ in range(warmup):
        fn()
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        runs.append((time.perf_counter() - t0) / iters)
    return statistics.median(runs)


# ---------- A) inspect the autograd graph of a small model ----------
model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)).eval()
x = torch.randn(256, 64)
y = model(x.requires_grad_())
print("input  shape/dtype/contiguous:", tuple(x.shape), x.dtype, x.is_contiguous())
print("output shape/dtype/contiguous:", tuple(y.shape), y.dtype, y.is_contiguous())
print("\nAutograd graph (walk grad_fn from output):")
def walk(fn, depth=0, seen=set()):
    if fn is None or fn in seen: return
    seen.add(fn); print("  " * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions: walk(nxt, depth + 1)
walk(y.grad_fn)
print("\nFX graph:"); print(torch.fx.symbolic_trace(model).graph)

# ---------- B) eager vs compiled / traced ----------
big = nn.Sequential(*[m for _ in range(6) for m in (nn.Linear(512, 512), nn.GELU())]).eval()
xb = torch.randn(128, 512)
with torch.no_grad():
    t_eager = bench(lambda: big(xb))
    print(f"\neager          : {t_eager*1e3:.3f} ms/call")
    try:
        cm = torch.compile(big)
        t0 = time.perf_counter(); cm(xb); print(f"compile first call (compile cost): {time.perf_counter()-t0:.2f} s")
        t_c = bench(lambda: cm(xb)); print(f"torch.compile  : {t_c*1e3:.3f} ms/call  (speedup {t_eager/t_c:.2f}x)")
    except Exception as e:
        print("torch.compile unavailable here (", type(e).__name__, ") -> using jit.trace")
    tm = torch.jit.trace(big, xb)
    t_t = bench(lambda: tm(xb)); print(f"jit.trace      : {t_t*1e3:.3f} ms/call  (speedup {t_eager/t_t:.2f}x)")

# ---------- C) contiguous vs non-contiguous ----------
M = torch.randn(4096, 4096)
views = {"contiguous M": M, "transpose M.t() (non-contig)": M.t(), "strided M[:, ::2] (non-contig)": M[:, ::2]}
for n, v in views.items():
    print(f"\n{n}: shape={tuple(v.shape)} dtype={v.dtype} stride={v.stride()} contiguous={v.is_contiguous()}")
base = {}
for op_name, op in (("v * 2", lambda v: v * 2), ("v.sum(dim=1)", lambda v: v.sum(dim=1))):
    print(f"\noperation: {op_name}")
    for n, v in views.items():
        t = bench(lambda: op(v), warmup=3, iters=10)
        print(f"  {n:34s} {t*1e3:8.3f} ms")
tt = M.t()
print(f"\n.contiguous() copy cost: {bench(lambda: tt.contiguous(), 3, 10)*1e3:.3f} ms")