from model import Value
import random

class Module:
  def parameters(self):
    return []

  def zero_grad(self):
    for p in self.parameters(): p.grad = 0.0

class Neuron(Module):
  def __init__(self, nin):
    self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
    self.b = Value(random.uniform(-1.0, 1.0))

  def __call__(self, x):
    assert len(x) == len(self.w), f"input must match weight shape"
    return sum((wi * xi for wi, xi in zip(self.w, x)), self.b).tanh() # tanh(w * x + b)

  def parameters(self):
    return self.w + [self.b]

class Layer(Module):
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    a = [n(x) for n in self.neurons]
    return a[0] if len(a) == 1 else a

  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
  def __init__(self, nin, nouts):
    shapes = [nin] + nouts
    self.layers = [Layer(shapes[i], shapes[i+1]) for i in range(len(shapes) - 1)]

  def __call__(self, x):
    for l in self.layers: x = l(x)
    return x

  def parameters(self):
    return [p for l in self.layers for p in l.parameters()]