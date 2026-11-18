"""Tasks 1-3: a tiny scalar autograd engine (in the spirit of micrograd)."""
import math


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)          # the number itself
        self.grad = 0.0                  # d(output)/d(this node), filled by backward()
        self._parents = tuple(parents)   # nodes this one was computed from
        self._op = op                    # just a label, handy for debugging
        self._backward = lambda: None    # local derivative rule (set by each op)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ---------- Task 2: operations that build the graph ----------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():                      # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad             # += : ACCUMULATE, never overwrite
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():                      # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "exponent must be a plain number"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():                      # d(a^k)/da = k * a^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():                      # 1 if input > 0 else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # extra ops needed for the sigmoid + BCE model in Task 5
    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
        out = Value(s, (self,), "sigmoid")

        def _backward():                      # s * (1 - s)
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():                      # 1/x
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    # conveniences built from the ops above (they also build graph nodes)
    def __neg__(self):          return self * -1
    def __sub__(self, other):   return self + (-other if isinstance(other, Value) else -other)
    def __rsub__(self, other):  return (-self) + other
    def __radd__(self, other):  return self + other      # lets Python's sum() work
    def __rmul__(self, other):  return self * other
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    # ---------- Task 3: backward pass ----------
    def backward(self):
        # 1) topological sort (parents always come BEFORE children in `topo`)
        topo, visited, stack = [], set(), [(self, False)]
        while stack:                          # iterative DFS -> no recursion limit issues
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                stack.append((p, False))
        # 2) seed: d(out)/d(out) = 1
        self.grad = 1.0
        # 3) visit in reverse order, each node pushes gradient to its parents
        for node in reversed(topo):
            node._backward()