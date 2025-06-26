#!/usr/bin/env python3

# ----------------------------------------------

# from model import Value

# x1 = Value(2.0, label="x1")
# x2 = Value(0.0, label="x2")
# w1 = Value(-3.0, label="w1")
# w2 = Value(1.0, label="w2")
# b = Value(6.8813735870195432, label="b")
# # x1*w1 + x2*w2 + b
# x1w1 = x1*w1; x1w1.label = "x1*w1"
# x2w2 = x2*w2; x2w2.label = "x2*w2"
# x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1*w1 + x2*w2"
# n = x1w1x2w2 + b; n.label = "n"

# o = n.tanh()
# e = (2*n).exp()
# o = (e - 1) / (e + 1)
# o.label = "o"

# o.backward()
# o.graph().render("o")

# ----------------------------------------------

from nn import MLP
import random

# random.seed(1)

xs = [
    [2.0,  3.0, -1.0],
    [3.0, -1.0,  0.5],
    [0.5,  1.0,  1.0],
    [1.0,  1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]
n = MLP(3, [4, 4, 1])

for i in range(100):
  # forward pass
  ypred = [n(x) for x in xs]
  loss = sum((yp - yr) ** 2 for yp, yr in zip(ypred, ys))
  # print(ypred)
  print(f"{i:4d}   {loss.value:.12f}")

  # backward pass
  n.zero_grad()
  loss.backward()

  # print(n.layers[0].neurons[0].w[0])

  for p in n.parameters():
    p.value += 0.001 * p.grad

print(ypred)