#!/usr/bin/env python3

from tinygrad import Tensor, dtypes, nn

# ****************************** hyper parameters ******************************
batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ******************************************************************************

# set manual seed to reproducability
Tensor.manual_seed(69420)

with open("input.txt", "r") as f: text = f.read()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
ctoi, itoc = {}, {}
for i, c in enumerate(vocab): ctoi[c], itoc[i] = i, c
encode = lambda x: [ctoi[c] for c in x]
decode = lambda x: "".join(itoc[i] for i in x)

data = Tensor(encode(text), dtype=dtypes.long)
# split data into train and validation chunks
train_data_sz = int(0.9 * len(data))
train_data = data[:train_data_sz]
valid_data = data[train_data_sz:]

def get_batch(split):
  data = train_data if split == "train" else valid_data
  ix = Tensor.randint(batch_size, low=0, high=len(data) - block_size).view(-1, 1) + Tensor.arange(0, block_size).view(1, -1)
  return data[ix], data[ix+1]

class Head:
  def __init__(self, head_size):
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.tril = Tensor.ones(block_size, block_size).tril()
  def __call__(self, x): return self.forward(x)
  def forward(self, x: Tensor):
    _, T, C = x.shape
    wei = self.query(x) @ self.key(x).transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")).softmax().dropout()
    return wei @ self.value(x)
  def params(self): return [self.key.weight, self.query.weight, self.value.weight]

class MultiHeadAttention:
  def __init__(self, num_heads, head_size):
    self.heads = [Head(head_size) for _ in range(num_heads)]
    self.proj = nn.Linear(n_embd, n_embd)
  def __call__(self, x): return self.forward(x)
  def forward(self, x): return self.proj(Tensor.cat([h(x) for h in self.heads], dim=-1)).dropout()
  def params(self): return [p for h in self.heads for p in h.params()] + [self.proj.weight, self.proj.bias]

class FeedForward:
  def __init__(self, n_embd):
    self.l1 = nn.Linear(n_embd, 4 * n_embd)
    self.l2 = nn.Linear(4 * n_embd, n_embd)
  def __call__(self, x): return self.forward(x)
  def forward(self, x): return self.l2(self.l1(x).relu()).dropout()
  def params(self): return [self.l1.weight, self.l1.bias, self.l2.weight, self.l2.bias]

class Block:
  def __init__(self, n_embd, n_head):
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  def __call__(self, x): return self.forward(x)
  def forward(self, x): return self.ffwd(self.ln2(self.sa(self.ln1(x)) + x)) + x
  def params(self): return self.sa.params() + self.ffwd.params() + [self.ln1.weight, self.ln1.bias, self.ln2.weight, self.ln2.bias]

class BigramLanguageModel:
  def __init__(self):
    self.token_emb = nn.Embedding(vocab_size, n_embd)
    self.pos_emb = nn.Embedding(block_size, n_embd)
    self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def __call__(self, idx, targets=None): return self.forward(idx, targets)

  def params(self):
    return [self.token_emb.weight, self.pos_emb.weight, self.ln_f.weight, self.ln_f.bias, self.lm_head.weight, self.lm_head.bias] + [p for b in self.blocks for p in b.params()]

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_emb, pos_emb = self.token_emb(idx), self.pos_emb(Tensor.arange(0, T))
    x = tok_emb + pos_emb
    for b in self.blocks: x = b(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      loss = logits.view(B*T, C).cross_entropy(targets.view(B*T))

    return logits, loss

model = BigramLanguageModel()
n_params = sum(p.numel() for p in model.params())
print(f"{n_params} parameters")

optimizer = nn.optim.AdamW(model.params(), lr=learning_rate)

with Tensor.train():
  for i in range(max_iters):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % eval_interval == 0 or i == max_iters - 1: print(f"loss: {loss.item():.8f}")