#!/usr/bin/env python3

import random
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

with open("names.txt", "r") as f: words = f.read().splitlines()
chars = ["."] + sorted(set(c for w in words for c in w))
ctoi = { s: i for i, s in enumerate(chars) }

BATCH_SIZE = 32
CHAR_CNT = len(chars)
CTX_SIZE = 5
FEATURE_SIZE = 10
HIDDEN_SIZE = 200

def build_dataset(words):
  x, y = [], []
  for word in words:
    ctx = [0] * CTX_SIZE
    for c in word + ".":
      # print("".join(map(lambda a: chars[a], w)), "->", c)
      x.append(ctx)
      y.append(ctoi[c])
      ctx = ctx[1:] + [ctoi[c]]
  return torch.tensor(x), torch.tensor(y)

# split the dataset into training (80%), development (10%) and testing (10%)
random.seed(1234567890)
random.shuffle(words)

n1, n2 = int(0.80 * len(words)), int(0.90 * len(words))

X_train, Y_train = build_dataset(words[:n1])
X_dev, Y_dev = build_dataset(words[n1:n2])
X_test, Y_test = build_dataset(words[n2:])

# X has shape (number of examples, context size)
# row i of X represents the character context for the ith example
# print(f"{X.shape=}")
# Y has shape (number of examples)
# element i of Y represents the expected character given the character context of the ith example
# print(f"{Y.shape=}")

# random number generator with fixed seed for reproducability
g = torch.Generator().manual_seed(2147483647)

# randomly generate the context to feature vector mapping matrix
C = torch.randn((CHAR_CNT, FEATURE_SIZE), requires_grad=True, generator=g)
W1 = torch.randn((CTX_SIZE * FEATURE_SIZE, HIDDEN_SIZE), requires_grad=True, generator=g)
b1 = torch.randn(HIDDEN_SIZE, requires_grad=True, generator=g)
W2 = torch.randn((HIDDEN_SIZE, CHAR_CNT), requires_grad=True, generator=g)
b2 = torch.randn(CHAR_CNT, requires_grad=True, generator=g)

params = [C, W1, b1, W2, b2]
nparams = sum(p.nelement() for p in params)
print(f"number of parameters: {nparams}")

stepi, lri, lossi = [], [], []

for i in range(200000):
  ix = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))

  # forward pass
  emb = C[X_train[ix]]
  h = torch.tanh(emb.view(-1, W1.shape[0]) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Y_train[ix])

  # backward pass
  for p in params: p.grad = None
  loss.backward()

  lr = 0.1 if i < 100000 else 0.01
  for p in params: p.data += -lr * p.grad

  stepi.append(i)
  # lri.append(lr)
  lossi.append(loss.item())

emb = C[X_train]
h = torch.tanh(emb.view(-1, W1.shape[0]) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y_train)

emb = C[X_dev]
h = torch.tanh(emb.view(-1, W1.shape[0]) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y_dev)

print(f"training loss: {loss.item():.6f}")
print(f"     dev loss: {loss.item():.6f}")

# plt.plot(stepi, lossi)
# plt.tight_layout()
# plt.show()

# for i in range(10):
#   out = []
#   ctx = [0] * CTX_SIZE
#   while True:
#     emb = C[ctx]
#     h = torch.tanh(emb.view(-1, W1.shape[0]) @ W1 + b1)
#     logits = h @ W2 + b2
#     probs = logits.softmax(1)
#     ix = probs.multinomial(1, replacement=True).item()
#     if ix == 0: break
#     ctx = ctx[1:] + [ix]
#     out.append(chars[ix])
#   print("".join(out))