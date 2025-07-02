#!/usr/bin/env python3

import random
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

class MLP:
  def __init__(self, words, n_ctx, n_embed, n_hidden):
    self.vocab = ["."] + sorted(set(c for w in words for c in w))
    self.itoc = { i: c for i, c in enumerate(self.vocab) }
    self.ctoi = { c: i for i, c in enumerate(self.vocab) }
    self.n_vocab = len(self.vocab)
    self.n_ctx = n_ctx
    self.n_embed = n_embed
    self.n_hidden = n_hidden

    # random number generator with fixed seed for reproducability
    # g = torch.Generator().manual_seed(2147483647)
    g = None
    self.c = torch.randn((self.n_vocab, n_embed), generator=g)
    self.w1 = torch.randn((n_ctx * n_embed, n_hidden), generator=g) * (5 / 3) / ((n_ctx * n_embed) ** 0.5)
    self.w2 = torch.randn((n_hidden, self.n_vocab), generator=g) * 0.01
    self.b2 = torch.randn(self.n_vocab, generator=g) * 0.0
    self.bngain = torch.zeros((1, n_hidden))
    self.bnbias = torch.ones((1, n_hidden))
    self.bnmean = torch.zeros((1, n_hidden))
    self.bnstd = torch.ones((1, n_hidden))
    self.parameters = [self.c, self.w1, self.w2, self.b2, self.bngain, self.bnbias]
    self.n_parameters = sum(p.nelement() for p in self.parameters)
    for p in self.parameters: p.requires_grad = True

    # random.seed(1234567890)
    random.shuffle(words)
    n1, n2 = int(0.80 * len(words)), int(0.90 * len(words))
    self.data = { "train": self._build_dataset(words[:n1]), "valid": self._build_dataset(words[n1:n2]), "test": self._build_dataset(words[n2:]) }

  def _build_dataset(self, words):
    x, y = [], []
    for word in words:
      ctx = [0] * self.n_ctx
      for c in word + ".":
        x.append(ctx)
        y.append(self.ctoi[c])
        ctx = ctx[1:] + [self.ctoi[c]]
    return torch.tensor(x), torch.tensor(y)

  def __call__(self, x, y, bs=-1):
    if bs > 0: x, y = x[ix := torch.randint(0, x.shape[0], (bs,))], y[ix]

    # forward pass
    emb = self.c[x]
    hpreact = emb.view(-1, self.w1.shape[0]) @ self.w1

    # batch normalization
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi = hpreact.std(0, keepdim=True)
    hnorm = self.bngain * (hpreact - bnmeani) / bnstdi + self.bnbias
    with torch.no_grad():
      self.bnmean = 0.999 * self.bnmean + 0.001 * bnmeani
      self.bnstd = 0.999 * self.bnstd + 0.001 * bnstdi

    h = hnorm.tanh()
    logits = h @ self.w2 + self.b2
    loss = F.cross_entropy(logits, y)

    return logits, loss

  def step(self, lr, bs):
    _, loss = self(*self.data["train"], bs=bs)

    # backward pass
    for p in self.parameters: p.grad = None
    loss.backward()
    for p in self.parameters: p.data += -lr * p.grad

    return loss

  @torch.no_grad()
  def sample(self, count):
    samples = []
    for _ in range(count):
      ctx, word = [0] * self.n_ctx, []
      while True:
        emb = self.c[ctx]
        hpreact = emb.view(-1, self.w1.shape[0]) @ self.w1
        hnorm = self.bngain * (hpreact - self.bnmean) / self.bnstd + self.bnbias
        h = hnorm.tanh()
        logits = h @ self.w2 + self.b2
        probs = logits.softmax(1)
        ix = probs.multinomial(1, replacement=True).item()
        if ix == 0: break
        ctx = ctx[1:] + [ix]
        word.append(self.itoc[ix])
      samples.append("".join(word))
    return samples

if __name__ == "__main__":
  with open("names.txt", "r") as f: words = f.read().splitlines()

  n_batch = 32
  mlp = MLP(words, n_ctx=5, n_embed=30, n_hidden=300)
  # mlp = MLP(words, n_ctx=3, n_embed=10, n_hidden=200)
  print(f"number of parameters: {mlp.n_parameters}")
  print()

  n_steps = 200000
  stepi = torch.arange(0, n_steps)
  lri = 10 ** -(stepi / 200000 + 1)
  lossi = []
  for i in range(n_steps):
    lr = lri[i]
    loss = mlp.step(lr, 64)
    if i % 10000 == 0: print(f"{i:6d}/{n_steps:6d}  batch size: {n_batch:4d}  loss: {loss.item():12.8f}")
    lossi.append(loss.log10().item())

  _, loss_train = mlp(*mlp.data["train"])
  _, loss_valid = mlp(*mlp.data["valid"])
  print()
  print(f"  training loss: {loss_train:.6f}")
  print(f"validation loss: {loss_valid:.6f}")

  print()
  print("\n".join(mlp.sample(30)))

  # plt.figure(figsize=(16, 9))
  # plt.plot(stepi, lossi, label="loss")

  # lr10000 = 10 ** -(stepi / 10000 + 1)
  # lr40000 = 10 ** -(stepi / 40000 + 1)
  # lr200000 = 10 ** -(stepi / 200000 + 1)
  # print(lr200000[-1])
  # plt.plot(stepi, lr10000, color="blue")
  # plt.plot(stepi, lr40000, color="red")
  # plt.plot(stepi, lr200000, color="green")

  # plt.legend(loc="upper right")
  # plt.tight_layout()
  # plt.show()