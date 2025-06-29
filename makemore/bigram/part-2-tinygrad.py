#!/usr/bin/env python3

# tinygrad must be installed with
#   python -m pip install -e . --config-settings editable_mode=strict
# so that vscode can find it
from tinygrad import Tensor
from os import getenv

SHOW_MAP = int(getenv("SHOW_MAP", 0))

with open("names.txt", "r") as f: words = f.read().splitlines()
chars = ["."] + sorted(set(c for w in words for c in w))
ctoi = { s: i for i, s in enumerate(chars) }

CHAR_CNT = len(chars)
N = [[1 for _ in range(CHAR_CNT)] for _ in range(CHAR_CNT)] # 1 instead of 0 to prevent having inf Negative Log Likelihood

# calculate the probabilities for each pair of characters from the training set
xs, ys = [], []
for word in words:
  word = "." + word + "."
  for c1, c2 in zip(word, word[1:]):
    xs.append(ctoi[c1])
    ys.append(ctoi[c2])

xs, ys = Tensor(xs), Tensor(ys)
xenc = xs.one_hot(CHAR_CNT)

if SHOW_MAP:
  from matplotlib import pyplot as plt
  plt.figure(figsize=(10, 10))
  plt.imshow(xenc.numpy())
  # plt.axis("off")
  plt.tight_layout()
  plt.show()

Tensor.manual_seed(2147483647) # adjust random seed for deterministic results
weights = Tensor.randn((CHAR_CNT, CHAR_CNT), requires_grad=True)

for _ in range(500):
  logits = xenc.dot(weights)
  # counts = logits.exp()
  # probs = counts / counts.sum(1, keepdim=True) # alternatively: probs = logits.softmax()
  probs = logits.softmax()
  loss = -probs[Tensor.arange(ys.size(0)), ys].log().mean()

  weights.grad = None
  loss.backward().realize()
  weights += -0.1 * weights.grad

  print(f"loss={loss.item():.6f}")

c = ctoi["."]
out = []
while True:
  probs = Tensor([c]).one_hot(CHAR_CNT).dot(weights).softmax()
  c = probs.multinomial(1, replacement=True).item()
  out.append(chars[c])
  if c == 0: break
print("".join(out))