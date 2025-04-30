import math
from graphviz import Digraph

class Value:
  def __init__(self, value, *, parents=(), op=None, label=""):
    self.value = value
    self.parents = set(parents)
    self.op = op
    self.label = label
    self.grad = 0.0
    self._backward = lambda: None

  def __repr__(self): return f"{self.label} = {self.value}"

  def __add__(self, o):
    out = Value(self.value + o.value, parents=(self, o), op="+", label=f"{self.label} + {o.label}")
    def bw(): self.grad, o.grad = out.grad, out.grad
    out._backward = bw
    return out

  def __mul__(self, o):
    out = Value(self.value * o.value, parents=(self, o), op="*", label=f"{self.label} * {o.label}")
    def bw(): self.grad, o.grad = o.value * out.grad, self.value * out.grad
    out._backward = bw
    return out

  def tanh(self):
    t = math.exp(2 * self.value)
    out = Value((t - 1) / (t + 1), parents=(self,), op="tanh", label=f"tanh({self.label})")
    def bw(): self.grad = 1 - out.value**2
    out._backward = bw
    return out

  def toposort(self):
    l = []
    v = set()
    def topo(e):
      if v not in v:
        v.add(e)
        for p in e.parents: topo(p)
        l.append(e)
    topo(self)
    return l

  def backward(self):
    self.grad = 1.0
    for e in reversed(self.toposort()): e._backward()

  def graph(self):
    dot = Digraph(format="pdf", graph_attr={"rankdir": "LR"})
    cnt, st = 0, [(0, self)]
    while len(st) > 0:
      child, e = st.pop()
      label = f"{{ {e.label} | value {e.value:.4f} | grad: {e.grad:.4f} }}"
      dot.node(name=str(cnt), label=label, shape="record")
      if child > 0: dot.edge(str(cnt), str(child))
      cnt += 1

      if e.op is not None:
        dot.node(name=str(cnt), label=e.op, shape="circle")
        dot.edge(str(cnt), str(cnt-1))
        cnt += 1

      st += [(cnt-1, p) for p in e.parents]
    return dot
