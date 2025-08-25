#!/usr/bin/env python3

import math
import time
import gzip
import struct
from pathlib import Path
from urllib import request
from tinygrad import Tensor, TinyJit, nn, dtypes
from matplotlib import pyplot as plt

Tensor.manual_seed(69420)

N_LABELS = 10
IMG_RES = 28
DATASET_URL = {
  "train-images": "https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz",
  "train-labels": "https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz",
  "test-images": "https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz",
  "test-labels": "https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz"
}

def get_data_raw(key: str):
  url = DATASET_URL[key]
  path = Path(__file__).resolve().parent / "data" / url.split("/")[-1]
  path.parent.mkdir(parents=True, exist_ok=True)
  if not path.exists():
    print(f"downloading {key} from {url} to {str(path)}")
    request.urlretrieve(url, str(path))
  return path

def load_images(path) -> Tensor:
  with gzip.open(path, "rb") as gz:
    assert struct.unpack("I", gz.read(4))[0] == 0x3080000, "invalid file format" # check magic number
    n, nrow, ncol = struct.unpack(">III", gz.read(12))
    assert nrow == ncol == IMG_RES, f"unexpected number of rows/columns ({nrow}/{ncol} != {IMG_RES})"
    return Tensor(gz.read(n * nrow * ncol)).view((n, nrow, ncol)).cast(dtypes.float16).realize()

def load_labels(path) -> Tensor:
  with gzip.open(path, "rb") as gz:
    assert struct.unpack("I", gz.read(4))[0] == 0x1080000, "invalid file format" # check magic number
    n = struct.unpack(">I", gz.read(4))[0]
    return Tensor(gz.read(n)).realize()

# ******************** MLP ********************

class MLPRaw:
  def __init__(self, n_hidden: int):
    bound1 = 1 / math.sqrt(IMG_RES**2)
    self.w1 = Tensor.uniform(IMG_RES ** 2, n_hidden, low=-bound1, high=bound1, requires_grad=True)
    self.b1 = Tensor.uniform(n_hidden, low=-bound1, high=bound1, requires_grad=True)
    bound2 = 1 / math.sqrt(n_hidden)
    self.w2 = Tensor.uniform(n_hidden, N_LABELS, low=-bound2, high=bound2, requires_grad=True)
    self.b2 = Tensor.uniform(N_LABELS, low=-bound2, high=bound2, requires_grad=True)

  def __call__(self, x: Tensor) -> Tensor:
    B, W, H = x.shape
    assert W == H == IMG_RES
    x = x.view(B, W * H)
    x = Tensor.tanh(x @ self.w1 + self.b1)
    x = Tensor.tanh(x @ self.w2 + self.b2)
    return x

class MLP:
  def __init__(self, n_hidden: int):
    self.l1 = nn.Linear(IMG_RES ** 2, n_hidden)
    self.l2 = nn.Linear(n_hidden, N_LABELS)
    self.params = nn.state.get_parameters(self)

  def __call__(self, x: Tensor) -> Tensor:
    B, W, H = x.shape
    assert W == H == IMG_RES
    x = x.view(B, W * H)
    x = self.l1(x).tanh()
    x = self.l2(x).tanh()
    return x

if __name__ == "__main__":
  X_train = (load_images(get_data_raw("train-images")) / 255.0).realize()
  Y_train = load_labels(get_data_raw("train-labels")).realize()
  X_test = (load_images(get_data_raw("test-images")) / 255.0).realize()
  Y_test = load_labels(get_data_raw("test-labels")).realize()

  # using continuous batching as random indexing is very slow
  # in production the data should be randomized first
  n_batch = 4096
  batch_idx = 0

  def get_batch(x: Tensor, y: Tensor):
    global batch_idx
    assert (n := len(x)) == len(y)
    start = batch_idx % n
    if start + n_batch >= n: start = 0
    batch_idx = (start + n_batch) % n
    return x[start:start+n_batch].contiguous(), y[start:start+n_batch].contiguous()

  # m = MLP(64)
  m = MLPRaw(64)
  params = nn.state.get_parameters(m)
  print(f"number of parameters: {sum(p.numel() for p in params)}")

  # using tinygrad's builtin optimizers as it's faster than manually updating the parameters
  lr_init = 0.005
  optimizer = nn.optim.AdamW(params, lr=lr_init)

  @TinyJit
  @Tensor.train()
  def step(x: Tensor, y: Tensor) -> Tensor:
    optimizer.zero_grad()
    logits = m(x)
    loss = logits.cross_entropy(y)
    loss.backward()
    return loss.realize(*optimizer.schedule_step())

  start = time.perf_counter()
  n_iter = 15000
  for i in range(n_iter):
    x, y = get_batch(X_train, Y_train)
    loss = step(x, y)
    if i % 1000 == 0: print(f"{i:10d} loss: {loss.item():.8f}")
  end = time.perf_counter()
  print(f"{end - start:.8f} seconds")

  train_loss = m(X_train).cross_entropy(Y_train).item()
  test_loss = m(X_test).cross_entropy(Y_test).item()

  print(f"train loss: {train_loss:.8f}")
  print(f"test loss: {test_loss:.8f}")

  n_gen = 40
  ix = Tensor.randint(n_gen, low=0, high=len(X_train))
  x, y = X_train[ix], Y_train[ix]

  logits = m(x)
  probs = logits.softmax()
  pred = probs.argmax(1).tolist()
  expected = y.tolist()

  print(pred)
  print(expected)

  ncol = 8
  nrow = math.ceil(n_gen / ncol)
  fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))

  for ax, img, p, e in zip(axs.flatten(), x, pred, expected):
    ax.axis("off")
    ax.set_title(f"expected: {e}; pred: {p}")
    ax.imshow(img.tolist(), cmap="gray" if e == p else "viridis")
  plt.tight_layout()
  plt.show()