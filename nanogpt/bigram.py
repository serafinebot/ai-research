#!/usr/bin/env python3

import torch
from torch import nn
import torch.nn.functional as F

with open("input.txt", "r") as f: text = f.read()

vocab = sorted(list(set(text)))
ctoi, itoc = {}, {}
for i, c in enumerate(vocab): ctoi[c], itoc[i] = i, c
encode = lambda x: [ctoi[c] for c in x]
decode = lambda x: "".join(itoc[i] for i in x)

data = torch.tensor(encode(text), dtype=torch.long)
# split data into train and validation chunks
n = int(0.9 * len(data))
train_data = data[:n]
valid_data = data[n:]

n_vocab = len(vocab) # number of characters in the vocabulary
n_batch = 32 # number of independent sequences will be processed in parallel
n_ctx = 8 # maximum number of characters used to predict the next one

# set manual seed to make output deterministic for testing
torch.manual_seed(69420)

def get_batch(data, bsize):
  ix = torch.randint(0, n_vocab, (bsize,))
  x = torch.stack([data[i:i+n_ctx] for i in ix])
  y = torch.stack([data[i+1:i+n_ctx+1] for i in ix])
  return x, y

xb, yb = get_batch(train_data, n_batch)

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(n_vocab, n_vocab)

  def forward(self, x, y=None):
    # x.shape = (n_batch, n_ctx); 
    # logits.shape = (n_batch, n_ctx, n_vocab)
    logits = self.token_embedding_table(x)
    loss = None if y is None else F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    return logits, loss

  def generate(self, x, max_tokens):
    for _ in range(max_tokens):
      logits, _ = self.forward(x)
      logits = logits[:, -1, :] # focus only on the last time step
      probs = F.softmax(logits, dim=-1)
      x_next = torch.multinomial(probs, 1)
      x = torch.cat((x, x_next), dim=1)
    return x

model = BigramLanguageModel()
logits, loss = model(xb, yb)
print(f"pre training loss: {loss.item():.8f}")
print("pre training generation: ")
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_tokens=300)[0].tolist()))
print()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for i in range(10000):
  xb, yb = get_batch(train_data, n_batch)
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  if i % 1000 == 0:
    valid_loss = model(*get_batch(valid_data, len(valid_data)))[1]
    print(f"{i:>8d} train: {loss.item():.8f} | valid: {valid_loss.item():.8f}")

# train_loss = model(*get_batch(train_data, len(train_data)))[1]
valid_loss = model(*get_batch(valid_data, len(valid_data)))[1]
# print(f"post training loss (train): {train_loss.item():.8f}")
print(f"post training loss (valid): {valid_loss.item():.8f}")
print("pre training generation: ")
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_tokens=300)[0].tolist()))
print()