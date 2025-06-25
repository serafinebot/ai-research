import math
import hashlib
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
    o = o if isinstance(o, Value) else Value(o, label=str(o))
    out = Value(self.value + o.value, parents=(self, o), op="+", label=f"{self.label} + {o.label}")
    def bw():
      self.grad += out.grad
      o.grad += out.grad
    out._backward = bw
    return out

  def __radd__(self, o):
    return self + o

  def __neg__(self):
    return self * -1

  def __sub__(self, o):
    return self + (-o)

  def __mul__(self, o):
    o = o if isinstance(o, Value) else Value(o, label=str(o))
    out = Value(self.value * o.value, parents=(self, o), op="*", label=f"{self.label} * {o.label}")
    def bw():
      self.grad += o.value * out.grad
      o.grad += self.value * out.grad
    out._backward = bw
    return out

  def __rmul__(self, o):
    return self * o

  def __truediv__(self, o):
    return self * o ** -1

  def exp(self):
    out = Value(math.exp(self.value), parents=(self,), op="exp", label=f"e^{self.label}")
    def bw():
      self.grad = out.grad * out.value
    out._backward = bw
    return out

  def __pow__(self, o):
    assert isinstance(o, (int, float)), "only int and float are supported"
    out = Value(self.value ** o, parents=(self,), op=f"^{o}", label=f"{self.label}^{o}")
    def bw():
      self.grad += o * self.value ** (o - 1) * out.grad
    out._backward = bw
    return out

  def __rpow__(self, o):
    return self ** o

  def tanh(self):
    t = math.exp(2 * self.value)
    out = Value((t - 1) / (t + 1), parents=(self,), op="tanh", label=f"tanh({self.label})")
    def bw():
      self.grad += 1 - out.value**2
    out._backward = bw
    return out

  def toposort(self):
    l = []
    v = set()
    def topo(e):
      if e not in v:
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
    nodes, st = set(), [self]
    while len(st) > 0:
      node = st.pop()
      if (i := id(node)) in nodes: continue
      nodes.add(i)
      label = f"{{ {node.label} | value {node.value:.4f} | grad: {node.grad:.4f} }}"
      dot.node(name=str(i), label=label, shape="record")
      if node.op is None: continue
      opid = f"{i}{node.op}"
      dot.node(name=opid, label=node.op, shape="circle")
      dot.edge(str(opid), str(i))
      for p in node.parents:
        j = id(p)
        dot.edge(str(j), str(opid))
        st.append(p)
    return dot