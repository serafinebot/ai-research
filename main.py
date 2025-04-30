#!/usr/bin/env python3

from __future__ import annotations

from model import Value

# a = Value(2.0, label="a")
# b = Value(-3.0, label="b")
# c = Value(10.0, label="c")
# e = a * b; e.label = "e"
# d = e + c; d.label = "d"
# f = Value(-2.0, label="f")
# L = d * f; L.label = "L"
#
# L.backward()
# L.graph().render("L")

x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
b = Value(6.8813735870195432, label="b")
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = "x1*w1"
x2w2 = x2*w2; x2w2.label = "x2*w2"
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b; n.label = "n"
o = n.tanh(); o.label = "o"

o.backward()
o.graph().render("o")
