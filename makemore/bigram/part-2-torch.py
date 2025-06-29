#!/usr/bin/env python3

from os import getenv
import torch.nn.functional as F
import torch

SHOW_MAP = int(getenv("SHOW_MAP", 0))

with open("names.txt", "r") as f: words = f.read().splitlines()
chars = ["."] + sorted(set(c for w in words for c in w))
ctoi = { s: i for i, s in enumerate(chars) }
CHAR_CNT = len(chars)

# calculate the probabilities for each pair of characters from the training set
xs, ys = [], []
for word in words:
  word = "." + word + "."
  for c1, c2 in zip(word, word[1:]):
    xs.append(ctoi[c1])
    ys.append(ctoi[c2])

xs, ys = torch.tensor(xs), torch.tensor(ys)
num = xs.nelement()
xenc = F.one_hot(xs, num_classes=CHAR_CNT).float()

if SHOW_MAP:
  from matplotlib import pyplot as plt
  plt.figure(figsize=(10, 10))
  plt.imshow(xenc.numpy())
  # plt.axis("off")
  plt.tight_layout()
  plt.show()

g = torch.Generator().manual_seed(2147483647)
weights = torch.randn((CHAR_CNT, CHAR_CNT), generator=g, requires_grad=True)

for i in range(100):
  logits = xenc @ weights
  counts = logits.exp()
  probs = counts / counts.sum(1, keepdim=True)
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (weights**2).mean()
  print(f"iter {i:5d}  loss {loss.item():12.6f}")

  weights.grad = None
  loss.backward()

  weights.data += -50 * weights.grad

for i in range(20):
  out = []
  ix = 0
  while True:
    logits = F.one_hot(torch.tensor([ix]), num_classes=CHAR_CNT).float() @ weights
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    ix = probs.multinomial(1, replacement=True, generator=g).item()
    out.append(chars[ix])
    if ix == 0: break
  print("".join(out))