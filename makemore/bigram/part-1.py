#!/usr/bin/env python3

# tinygrad must be installed with
#   python -m pip install -e . --config-settings editable_mode=strict
# so that vscode can find it
from tinygrad import Tensor

SHOW_MAP = False

with open("names.txt", "r") as f: words = f.read().splitlines()
chars = ["."] + sorted(set(c for w in words for c in w))
ctoi = { s: i for i, s in enumerate(chars) }

CHAR_CNT = len(chars)
N = [[1 for _ in range(CHAR_CNT)] for _ in range(CHAR_CNT)] # 1 instead of 0 to prevent having inf Negative Log Likelihood

# calculate the probabilities for each pair of characters from the training set
for word in words:
  word = "." + word + "."
  for c1, c2 in zip(word, word[1:]):
    N[ctoi[c1]][ctoi[c2]] += 1
N = Tensor(N, dtype="int32")
P = N.cast("float32") / N.sum(1, keepdim=True)

if SHOW_MAP:
  from matplotlib import pyplot as plt
  plt.figure(figsize=(10, 10))
  plt.imshow(N.numpy(), cmap="Blues")
  for i in range(CHAR_CNT):
    for j in range(CHAR_CNT):
      plt.text(j, i, chars[i] + chars[j], ha="center", va="bottom", color="gray", size=8)
      plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray", size=8)
  plt.axis("off")
  plt.tight_layout()
  plt.show()

# TODO: this section is slow because tinygrad runs the multinomial kernel everytime (.item() for each while loop iteration; and multiple kernels per loop?), find a better way to do this
# predict some words
Tensor.manual_seed(2147483647) # adjust random seed for deterministic results
for _ in range(5):
  i = 0
  out = []
  while True:
    p = P[i]
    i = p.multinomial(1, replacement=True).item()
    out.append(chars[i])
    if i == 0: break
  print("".join(out))
print()

# calculate the negative log likelihood of the model
L = P.log().tolist()
n, ll = 0, 0
for word in words:
  word = "." + word + "."
  for c1, c2 in zip(word, word[1:]):
    ll -= L[ctoi[c1]][ctoi[c2]]
    n += 1
nll = ll / n
print(f"{nll=}")