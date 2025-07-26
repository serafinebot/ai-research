#!/usr/bin/env python3

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from colors import Colors

if __name__ == "__main__":
  # ********************** BUILD DATASET ************************
  with open("names.txt", "r") as f: words = f.read().splitlines()

  vocab = ["."] + sorted(set(c for w in words for c in w))
  itoc = { i: c for i, c in enumerate(vocab) }
  ctoi = { c: i for i, c in enumerate(vocab) }

  n_ctx = 3
  n_embed = 10
  n_hidden = 64
  n_vocab = len(vocab)
  n_batch = 32

  def build_dataset(words):
    x, y = [], []
    for word in words:
      ctx = [0] * n_ctx
      for c in word + ".":
        x.append(ctx)
        y.append(ctoi[c])
        ctx = ctx[1:] + [ctoi[c]]
    return torch.tensor(x), torch.tensor(y)

  random.seed(1234567890)
  random.shuffle(words)
  n1, n2 = int(0.80 * len(words)), int(0.90 * len(words))
  data = { "train": build_dataset(words[:n1]), "valid": build_dataset(words[n1:n2]), "test": build_dataset(words[n2:]) }


  # ********************** SETUP PARAMETERS ************************
  g = torch.Generator().manual_seed(2147483647)
  # embed
  C = torch.randn((n_vocab, n_embed), generator=g)
  # layer 1
  w1 = torch.randn((n_embed * n_ctx, n_hidden), generator=g) * (5/3) / ((n_embed * n_ctx) ** 0.5)
  # b1 = torch.randn(n_hidden, generator=g) * 0.1 # not needed as it is useless because of batch norm
  # layer 2
  w2 = torch.randn((n_hidden, n_vocab), generator=g) * 0.1
  b2 = torch.randn(n_vocab, generator=g) * 0.1
  # batch norm
  bngain = torch.randn(n_hidden, generator=g) * 0.1 + 1.0
  bnbias = torch.randn(n_hidden, generator=g) * 0.1

  parameters = [C, w1, w2, b2, bngain, bnbias]
  n_parameters = sum(p.nelement() for p in parameters)
  for p in parameters: p.requires_grad = True
  print(f"number of parameters: {n_parameters}")

  # get training data in a batch for a single forward pass
  Xtr, Ytr = data["train"]
  ix = torch.randint(0, Xtr.shape[0], (n_batch,), generator=g)
  X, Y = Xtr[ix], Ytr[ix]

  # print(f"{X.shape=}")
  # print(f"{Y.shape=}")
  # print(X)
  # print(Y)


  # ********************** FORWARD PASS IN CHUNKS ************************
  emb = C[X] # embed the characters into vectors
  embcat = emb.view(emb.shape[0], -1) # for each batch, concatenate all of the character embeddings

  # linear layer 1
  hprebn = embcat @ w1 # hidden layer pre-activation
  # batchnorm layer
  bnmeani = 1.0 / n_batch * hprebn.sum(0, keepdim=True) # isn't keepdim redundant?
  bndiff = hprebn - bnmeani
  bndiff2 = bndiff ** 2
  bnvar = 1.0 / (n_batch - 1) * bndiff2.sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1 instead of n)
  bnvar_inv = (bnvar + 1e-5) ** -0.5
  bnraw = bndiff * bnvar_inv
  hpreact = bngain * bnraw + bnbias
  # non-linearity
  h = hpreact.tanh()
  # linear layer 2
  logits = h @ w2 + b2
  # cross entropy loss
  logit_maxes, logit_idx = logits.max(1, keepdim=True)
  norm_logits = logits - logit_maxes # subtract max for numerical stability
  counts = norm_logits.exp()
  counts_sum = counts.sum(1, keepdim=True)
  counts_sum_inv = counts_sum ** -1
  probs = counts * counts_sum_inv
  logprobs = probs.log()
  loss = -logprobs[range(n_batch), Y].mean()

  # pytorch backward pass
  for p in parameters: p.grad = None
  for t in (loss, logprobs, probs, counts_sum_inv, counts_sum, counts, norm_logits, logit_maxes, logits,
            h, hpreact, bnraw, bnvar_inv, bnvar, bndiff2, bndiff, bnmeani, hprebn, embcat, emb):
    t.retain_grad()
  loss.backward()
  print(f"loss: {loss.item():.6f}")


  # ********************** MANUAL BACKPROP IN CHUNKS ************************
  def cmp(name, dt, t):
    exact = torch.all(t.grad == dt).item()
    approx = torch.allclose(t.grad, dt)
    maxdiff = (t.grad - dt).abs().max().item()
    print(f"{Colors.GREEN if exact or approx else Colors.RED} {name:>15s} {Colors.RESET} maxdiff: {maxdiff:.6f} {"*" if not exact and approx else ""}")

  # loss = -logprobs[range(n_batch), Y].mean()
  # mean ---> (X1 + ... + Xn) / n = (X1 + ... + Xn) * n**-1 = n**-1 * X1 + ... + n**-1 * Xn
  # d/dX1 [n**-1 * X1 + ... + n**-1 + Xn] = n**-1 + 0 + ... + 0 = n**-1
  dlogprobs = torch.zeros_like(logprobs)
  dlogprobs[range(n_batch), Y] = -n_batch ** -1
  cmp("logprobs", dlogprobs, logprobs)

  # logprobs = probs.log()
  # b = log(a) ---> db/da = d/da [log(a)] = 1/a = a ** -1
  # a is probs and we have to multiply it by dlogprobs because of the chain rule
  dprobs = dlogprobs * probs ** -1
  cmp("probs", dprobs, probs)

  # probs = counts * counts_sum_inv
  # c = a * b, where a[3x3] and b[3x1]
  # 
  #                             -> broadcasting ->                                    
  #    | a1,1   a1,2   a1,3 |   | b1 |       | a1,1   a1,2   a1,3 |   | b1   b1   b1 |
  #    | a2,1   a2,2   a2,3 | * | b2 |   =   | a2,1   a2,2   a2,3 | * | b2   b2   b2 |
  #    | a3,1   a3,2   a3,3 |   | b3 |       | a3,1   a3,2   a3,3 |   | b3   b3   b3 |
  #
  #    | a1,1*b1   a1,2*b1   a1,3*b1 |
  #    | a2,1*b2   a2,2*b2   a2,3*b2 |
  #    | a3,1*b3   a3,2*b3   a3,3*b3 |

  # dc/db = d/db [a * b] = a
  # in this case a is counts and we have to multiply it by dprobs because of the chain rule
  # we have to sum the result accross dim 1 to join together all of the gradients from the different branches in the compute graph
  dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
  cmp("counts_sum_inv", dcounts_sum_inv, counts_sum_inv)

  # counts_sum_inv = counts_sum**-1
  # b = a**-1 -> db/da = d/da [a**-1] = -1*a**-2
  dcounts_sum = dcounts_sum_inv * -counts_sum**-2
  cmp("counts_sum", dcounts_sum, counts_sum)

  # probs = counts * counts_sum_inv (same as dcounts_sum_inv)
  dcounts = counts_sum_inv * dprobs
  # counts_sum = counts.sum(1, keepdim=True)
  # summation is just a gradient broadcast
  #    | b1 |       | a1,1 + a1,2 + a1,3 |
  #    | b2 |   =   | a2,1 + a2,2 + a2,3 |
  #    | b3 |       | a3,1 + a3,2 + a3,3 |
  # the derivative for each b with respect to each element of the same row of matrix a is 1
  #    b = a1 + a2 + a3
  #    db/da1 = d/da1 [a1 + a2 + a3] = 1
  #    db/da2 = d/da2 [a1 + a2 + a3] = 1
  #    db/da3 = d/da3 [a1 + a2 + a3] = 1
  # because of the chain rule, we multiply the derivative of the outer function. so this step (summation) is just broadcasting
  # the derivative of the outer function to each element of a
  # we have to sum the result to dcounts because we are merging two branches in the compute graph
  dcounts += torch.ones_like(counts) * dcounts_sum
  cmp("counts", dcounts, counts)

  # counts = norm_logits.exp()
  # the derivative of e^a is e^a
  #   b = e^a ---> db/da = d/da [e^a] = e^a
  # dnorm_logits = dcounts * norm_logits.exp() = dcounts * counts
  dnorm_logits = dcounts * counts
  cmp("norm_logits", dnorm_logits, norm_logits)

  # norm_logits = logits - logit_maxes
  #    | c1,1   c1,2   c1,3 |       | a1,1   a1,2   a1,3 |       | b1 |
  #    | c2,1   c2,2   c2,3 |   =   | a2,1   a2,2   a2,3 |   -   | b2 |
  #    | c3,1   c3,2   c3,3 |       | a3,1   a3,2   a3,3 |       | b3 |
  # c1,1 = a1,1 - b1 ---> dc/db = -1
  #                       dc/da = 1

  dlogit_maxes = -dnorm_logits.sum(1, keepdim=True)
  cmp("logit_maxes", dlogit_maxes, logit_maxes)

  dlogits = dnorm_logits.clone()
  dlogits += F.one_hot(logit_idx.squeeze(), dlogits.shape[1]) * dlogit_maxes
  cmp("logits", dlogits, logits)

  # logits = h @ w2 + b2
  # the backward pass of a matrix multiplication turns out to be another matrix multiplication
  dh = dlogits @ w2.T
  cmp("h", dh, h)

  dw2 = h.T @ dlogits
  cmp("w2", dw2, w2)

  db2 = dlogits.sum(0)
  cmp("db2", db2, b2)

  # h = hpreact.tanh()
  # b = tanh(a)
  # db/da = d/da [tanh(a)] = (e^a - e^(-a))/(e^a + e^(-a)) = 1 - b^2
  dhpreact = dh * (1.0 - h**2)
  cmp("hpreact", dhpreact, hpreact)

  # hpreact = bngain * bnraw + bnbias
  dbngain = (dhpreact * bnraw).sum(0)
  cmp("bngain", dbngain, bngain)

  dbnraw = dhpreact * bngain
  cmp("bnraw", dbnraw, bnraw)

  dbnbias = dhpreact.sum(0)
  cmp("bnbias", dbnbias, bnbias)

  # bnraw = bndiff * bnvar_inv
  dbnvar_inv = (dbnraw * bndiff).sum(0, keepdim=True)
  cmp("bnvar_inv", dbnvar_inv, bnvar_inv)

  dbndiff = dbnraw * bnvar_inv

  # bnvar_inv = (bnvar + 1e-5) ** -0.5
  # -0.5 * (x + 1e-5) ** -1.5
  dbnvar = dbnvar_inv * -0.5 * (bnvar + 1e-5) ** -1.5
  cmp("bnvar", dbnvar, bnvar)

  # bnvar = 1 / (n_batch - 1) * bndiff2.sum(0, keepdim=True)
  dbndiff2 = (1.0 / (n_batch - 1.0)) * torch.ones_like(bndiff2) * dbnvar
  cmp("bndiff2", dbndiff2, bndiff2)

  # bndiff2 = bndiff ** 2
  dbndiff += dbndiff2 * 2 * bndiff
  cmp("bndiff", dbndiff, bndiff)

  # bndiff = hprebn - bnmeani
  dhprebn = dbndiff.clone()

  dbnmeani = -dbndiff.sum(0, keepdim=True)
  cmp("bnmeani", dbnmeani, bnmeani)

  # bnmeani = 1/n_batch * hprebn.sum(0, keepdim=True)
  dhprebn += (1.0 / n_batch) * torch.ones_like(hprebn) * dbnmeani
  cmp("hprebn", dhprebn, hprebn)

  # hprebn = embcat @ w1
  dembcat = dhprebn @ w1.T
  cmp("embcat", dembcat, embcat)

  dw1 = embcat.T @ dhprebn
  cmp("w1", dw1, w1)

  demb = dembcat.view(emb.shape)
  cmp("emb", demb, emb)

  dC = torch.zeros_like(C)
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      dC[X[i,j]] += demb[i,j]
  cmp("C", dC, C)