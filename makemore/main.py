#!/usr/bin/env python3

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
from matplotlib import pyplot as plt

def s(shape): return "(" + ", ".join(map(str, shape)) + ")"

# ******************** MLP ********************

class MLP:
  def __init__(self, words, n_ctx, n_embed, n_hidden):
    self.vocab = ["."] + sorted(set(c for w in words for c in w))
    self.itoc = { i: c for i, c in enumerate(self.vocab) }
    self.ctoi = { c: i for i, c in enumerate(self.vocab) }
    self.n_vocab = len(self.vocab)
    self.n_ctx = n_ctx
    self.n_embed = n_embed
    self.n_hidden = n_hidden

    self.c = torch.randn((self.n_vocab, n_embed))
    self.w1 = torch.randn((n_ctx * n_embed, n_hidden)) * (5 / 3) / ((n_ctx * n_embed) ** 0.5)
    self.w2 = torch.randn((n_hidden, self.n_vocab)) * 0.01
    self.b2 = torch.randn(self.n_vocab) * 0.0
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

def test_mlp(words):
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


# ******************** MLP Torch ********************

class MLPTorch(nn.Module):
  def __init__(self, words, n_ctx, n_embed, n_hidden):
    super(MLPTorch, self).__init__()
    self.vocab = ["."] + sorted(set(c for w in words for c in w))
    self.itoc = { i: c for i, c in enumerate(self.vocab) }
    self.ctoi = { c: i for i, c in enumerate(self.vocab) }
    self.n_vocab = len(self.vocab)
    self.n_ctx = n_ctx
    self.n_embed = n_embed
    self.n_hidden = n_hidden

    self.c = torch.randn((self.n_vocab, n_embed))
    self.layers = nn.Sequential(
      nn.Linear(n_ctx * n_embed, n_hidden, bias=False), nn.BatchNorm1d(n_hidden), nn.Tanh(),
      nn.Linear(n_hidden, self.n_vocab), nn.BatchNorm1d(self.n_vocab)
    )

    with torch.no_grad():
      # make last layer less confident
      self.layers[-1].weight *= 0.1
      # apply gain to linear layers
      for layer in self.layers:
        if isinstance(layer, nn.Linear):
          layer.weight *= 1.0

    # self.parameters = [self.c] + [p for layer in self.layers for p in layer.parameters()]
    self.n_parameters = sum(p.nelement() for p in self.parameters())
    for p in self.parameters(): p.requires_grad = True

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

  def forward(self, x):
    emb = self.c[x]
    sz = emb.size()[:-2] + (-1,)
    if len(sz) == 1: sz = (1, sz[0])
    return self.layers(emb.view(*sz))

  def sample(self, n):
    tr = self.training
    self.eval()
    words = []
    for _ in range(n):
      ctx, word = [0] * self.n_ctx, []
      while True:
        probs = self(ctx).softmax(dim=1)
        ix = probs.multinomial(1, replacement=True).item()
        if ix == 0: break
        ctx = ctx[1:] + [ix]
        word.append(self.itoc[ix])
      words.append("".join(word))
    self.train(tr)
    return words

def test_mlp_torch(words):
  mlp = MLPTorch(words, n_ctx=5, n_embed=30, n_hidden=300)
  print(f"number of parameters: {mlp.n_parameters}")
  print()

  xtr, ytr = mlp.data["train"]
  xval, yval = mlp.data["valid"]

  mlp.train()
  n_batch = 64
  n_steps = 200000
  stepi = torch.arange(0, n_steps)
  lri = 10 ** -(stepi / 200000 + 1)
  lossi = []
  for i in range(n_steps):
    x, y = (xtr[ix := torch.randint(0, xtr.shape[0], (n_batch,))], ytr[ix]) if n_batch > 0 else (xtr, ytr)
    lr = lri[i]
    x = mlp(x)
    loss = F.cross_entropy(x, y)
    if i % 10000 == 0: print(f"{i:6d}/{n_steps:6d}  batch size: {n_batch:4d}  loss: {loss.item():12.8f}")
    lossi.append(loss.log10().item())

    for p in mlp.parameters(): p.grad = None
    loss.backward()
    for p in mlp.parameters(): p.data += -lr * p.grad

  mlp.eval()
  loss_train = F.cross_entropy(mlp(xtr), ytr)
  loss_valid = F.cross_entropy(mlp(xval), yval)
  print()
  print(f"  training loss: {loss_train:.6f}")
  print(f"validation loss: {loss_valid:.6f}")
  print()
  print("\n".join(mlp.sample(30)))


# ******************** CNN ********************

class MergeContiguous(nn.Module):
  def __init__(self, n):
    super(MergeContiguous, self).__init__()
    self.n = n

  def forward(self, x):
    return x.view(x.shape[0], x.shape[1] // self.n, x.shape[2] * self.n)

# custom BatchNorm1d as nn.BatchNorm1d expects the shape (N, C, L) and we are using (N, L, C)
class BatchNorm1d(nn.Module):
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    super(BatchNorm1d, self).__init__()
    self.eps = eps
    self.momentum = momentum
    self.training = True
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)

  def forward(self, x):
    if self.training:
      if x.ndim == 2: dim = 0
      elif x.ndim == 3: dim = (0, 1)
      xmean = x.mean(dim, keepdim=True)
      xvar = x.var(dim, keepdim=True)
    else:
      xmean = self.running_mean
      xvar = self.running_var

    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta

    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

class CNN(nn.Module):
  def __init__(self, words, n_ctx, n_embed, n_hidden):
    super(CNN, self).__init__()
    self.vocab = ["."] + sorted(set(c for w in words for c in w))
    self.itoc = { i: c for i, c in enumerate(self.vocab) }
    self.ctoi = { c: i for i, c in enumerate(self.vocab) }
    self.toc = lambda a: [self.itoc[b] for b in a] if isinstance(a, Iterable) else self.itoc[a]
    self.toi = lambda a: [self.ctoi[b] for b in a] if isinstance(a, Iterable) else self.ctoi[a]
    self.n_vocab = len(self.vocab)
    self.n_ctx = n_ctx
    self.n_embed = n_embed
    self.n_hidden = n_hidden

    # using LayerNorm over BatchNorm1d to avoid weird permutations
    # TODO: how different is LayerNorm from BatchNorm1d? What are the tradeoffs?
    self.layers = nn.Sequential(
      nn.Embedding(self.n_vocab, n_embed),
      MergeContiguous(2), nn.Linear(n_embed * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), nn.Tanh(),
      MergeContiguous(2), nn.Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), nn.Tanh(),
      MergeContiguous(2), nn.Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), nn.Tanh(),
      nn.Linear(n_hidden, self.n_vocab), nn.Flatten(start_dim=1)
    )

    with torch.no_grad():
      # make last layer less confident
      self.layers[-2].weight *= 0.1
      # apply gain to linear layers
      for layer in self.layers:
        if isinstance(layer, nn.Linear):
          layer.weight *= 1.0

    self.n_parameters = sum(p.nelement() for p in self.parameters())
    for p in self.parameters(): p.requires_grad = True

    # random.seed(1234567890) # set seed for reproducability
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

  def forward(self, x):
    # for l in self.layers:
    #   print(f"{s(x.shape):>12s} -> {l.__class__.__name__:16s} -> ", end="")
    #   x = l(x)
    #   print(f"{s(x.shape):>12s}")
    # return x
    return self.layers(x)

  def sample(self, n):
    tr = self.training
    self.eval()
    # TODO: as these are convolutions, is it possible to calculate all of the samples at once?
    words = []
    for _ in range(30):
      ctx = torch.zeros(1, self.n_ctx, dtype=torch.int)
      word = []
      while True:
        probs = self(ctx).softmax(dim=1)
        ix = probs.multinomial(1, replacement=True).item()
        if ix == 0: break
        ctx = ctx.roll(-1)
        ctx[-1] = ix
        word.append(self.itoc[ix])
      words.append("".join(word))
    self.train(tr)
    return words

def test_cnn(words):
  model = CNN(words, n_ctx=8, n_embed=24, n_hidden=300)
  print(f"number of parameters: {model.n_parameters}")
  print()

  xtr, ytr = model.data["train"]
  xval, yval = model.data["valid"]

  model.train()
  n_batch = 64
  n_steps = 200000
  n_steps = 10000
  # n_steps = 0
  stepi = torch.arange(0, n_steps)
  lri = 10 ** -(stepi / 200000 + 1)
  lossi = []
  for i in range(n_steps):
    x, y = (xtr[ix := torch.randint(0, xtr.shape[0], (n_batch,))], ytr[ix]) if n_batch > 0 else (xtr, ytr)
    # for wix, cix in zip(x, y): print(f"{"".join(model.toc(wix.tolist()))} -> {model.toc(cix.item())}")
    lr = lri[i]
    x = model(x)
    # print(x.shape)
    # return
    loss = F.cross_entropy(x, y)
    if i % 10000 == 0: print(f"{i:6d}/{n_steps:6d}  batch size: {n_batch:4d}  loss: {loss.item():12.8f}")
    lossi.append(loss.log10().item())

    for p in model.parameters(): p.grad = None
    loss.backward()
    for p in model.parameters(): p.data += -lr * p.grad

  model.eval()
  loss_train = F.cross_entropy(model(xtr), ytr)
  loss_valid = F.cross_entropy(model(xval), yval)
  print()
  print(f"  training loss: {loss_train:.6f}")
  print(f"validation loss: {loss_valid:.6f}")
  print()
  print("\n".join(model.sample(30)))

if __name__ == "__main__": 
  with open("names.txt", "r") as f: words = f.read().splitlines()
  # torch.manual_seed(1234567890)
  # test_mlp(words)
  # test_mlp_torch(words)
  test_cnn(words)