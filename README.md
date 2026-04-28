# VVTK Dataset

A high-performance sharded binary dataset library for PyTorch, with a C++ ring-buffered dataloader for zero-copy reads, multi-threaded prefetch, and automatic padding of variable-length data.

## Key Features

- **Sharded binary format** — data split across `.vvtk` shard files for parallel I/O
- **Zero-copy reads** — memory-mapped shards with direct tensor views
- **C++ dataloader** (`VVTKDataLoader`) — ring buffer with worker threads, batching, padding, shuffle
- **Mixed compression** — per-tensor modes: `none` (raw), `flac` (audio via [dr_flac](https://github.com/mackron/dr_libs)), `zstd` (general-purpose)
- **Variable-length support** — automatic padding to fixed shapes with real-length tracking
- **N-tuple samples** — each sample stores multiple tensors (e.g., audio + tokens)
- **Full dtype support** — uint8, int8, int16, int32, int64, float16, float32, float64

## Installation

**Requirements:** Python 3.8+, PyTorch 1.9+, C++17 compiler (GCC 7+)

```bash
git clone <repo-url>
cd vvtk_dataset
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

This builds the C++ extension and registers the package in your Python environment so you can `import vvtk_dataset` from anywhere.

**Verify:**

```bash
python tests.py
```

**Optional** (for FLAC / zstd compression):

```bash
pip install soundfile zstd
```
## Usage

### Writing a dataset

```python
from vvtk_dataset import VVTKDataset
import numpy as np

with VVTKDataset('data/train', mode='wb', num_shards=16,
                 compression=['none', 'none']) as ds:
    for i in range(50_000):
        image = np.random.randint(0, 256, (3, 32, 32), dtype=np.uint8)
        label = np.array([i % 100], dtype=np.int64)
        ds.add(i, image, label)
```

### Reading with PyTorch DataLoader

```python
from vvtk_dataset import VVTKDataset
from torch.utils.data import DataLoader

ds = VVTKDataset('data/train', mode='rb', compression=['none', 'none'])
loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)

for images, labels in loader:
    pass  # images: [128, 3, 32, 32], labels: [128, 1]
```

### Reading with VVTKDataLoader (C++ loader)

```python
from vvtk_dataset import VVTKDataset, VVTKDataLoader
import torch

ds = VVTKDataset('data/train', mode='rb', compression=['none', 'none'])

loader = VVTKDataLoader(
    ds,
    batch_size=128,
    num_workers=8,
    ring_size=4,
    shapes=[(3, 32, 32), (1,)],
    dtypes=[torch.uint8, torch.int64],
    shuffle=True
)

for batch in loader:
    (images, image_lengths), (labels, label_lengths) = batch
    pass  # images: [128, 3, 32, 32] uint8, pinned memory
```

### Variable-length audio with FLAC compression

```python
# Writing
with VVTKDataset('data/audio', mode='wb', num_shards=32,
                 compression=['flac', 'none'],
                 compression_args=[{'sample_rate': 16000}, {}]) as ds:
    for i in range(10_000):
        waveform = np.random.randn(n_samples).astype(np.float32)
        tokens = np.random.randint(0, 30000, size=tok_len, dtype=np.int16)
        ds.add(i, waveform, tokens)

# Reading — audio padded to 30s, tokens to 400
ds = VVTKDataset('data/audio', mode='rb',
                 compression=['flac', 'none'],
                 compression_args=[{'sample_rate': 16000}, {}])

loader = VVTKDataLoader(
    ds, batch_size=64, num_workers=8,
    shapes=[(480000,), (400,)],
    dtypes=[torch.float32, torch.int16],
    padding_values=[0.0, 0],
    shuffle=True
)

for batch in loader:
    (audio, audio_lengths), (tokens, token_lengths) = batch
    # audio: [64, 480000] float32 (zero-padded, real length in audio_lengths)
    pass
```

### Mixing variable-length and fixed-size tensors (`is_variable`)

Samples can mix **variable-length** tensors (padded to a max shape and
returned with a real length) and **fixed-size** tensors (returned as-is,
with no length). Use the `is_variable=[...]` flag, available on both
`VVTKDataset` and `VVTKDataLoader`, to mark which items are which.

Example: a fingerprint-verification sample with **6 tensors** — five
variable-length per-patch tensors and one scalar class label:

```python
from vvtk_dataset import VVTKDataset, VVTKDataLoader
import torch

MAX_N = 16   # max patches per sample

# Writing: each sample has N patches, N varies
with VVTKDataset('data/fp', mode='wb', num_shards=8,
                 compression=['none'] * 6) as ds:
    for i in range(50_000):
        n = np.random.randint(1, MAX_N + 1)
        ds.add(i,
               np.random.randn(n, 96, 96).astype(np.float32),  # x1: patches
               np.random.randn(n).astype(np.float32),          # x2
               np.random.randn(n).astype(np.float32),          # x3
               np.random.randn(n).astype(np.float32),          # x4
               np.random.randint(0, 10, n, dtype=np.int64),    # x5
               np.array([i % 100], dtype=np.int64))            # x6: label

# C++ loader — pad x1..x5, return x6 as a plain int64 scalar per sample
ds = VVTKDataset('data/fp', mode='rb', compression=['none'] * 6)

loader = VVTKDataLoader(
    ds, batch_size=32, num_workers=4, ring_size=4,
    shapes=[(MAX_N, 96, 96), (MAX_N,), (MAX_N,), (MAX_N,), (MAX_N,), (1,)],
    dtypes=[torch.float32, torch.float32, torch.float32, torch.float32,
            torch.int64,   torch.int64],
    padding_values=[0.0, 0.0, 0.0, 0.0, -1, 0],
    is_variable=[True, True, True, True, True, False],   # x6 is fixed
    shuffle=True,
)

for batch in loader:
    (x1, n1), (x2, n2), (x3, n3), (x4, n4), (x5, n5), label = batch
    # x1: [32, MAX_N, 96, 96] float32, n1: [32] int64 (real patch counts)
    # label: [32] int64 — no length companion
    pass
```

The same `is_variable=[...]` argument also applies to `VVTKDataset` for use
with the standard `torch.utils.data.DataLoader`.

See `examples/example61_fingerprint_multitensor/` for a runnable notebook.

### Mini-epochs with `nb_samples_per_epoch`

When training with large datasets you may want shorter epochs to report
metrics or save checkpoints more frequently.  `nb_samples_per_epoch` splits
the full dataset traversal into consecutive mini-epochs.  The data order is
only reshuffled once **all** samples have been seen.

```python
from vvtk_dataset import VVTKDataset, VVTKDataLoader
import torch

# Dataset: 1000 audio samples with token transcriptions
ds = VVTKDataset('data/audio', mode='rb',
                 compression=['flac', 'none'],
                 compression_args=[{'sample_rate': 16000}, {}])

loader = VVTKDataLoader(
    ds, batch_size=10, num_workers=4,
    shapes=[(480000,), (400,)],
    dtypes=[torch.float32, torch.int16],
    padding_values=[0.0, 0],
    shuffle=True,
    nb_samples_per_epoch=100      # 100 samples per mini-epoch (10 batches)
)

# 1000 samples / 100 per mini-epoch = 10 mini-epochs to see the full dataset.
# Running 20 epochs goes through the data exactly twice with a reshuffle
# after mini-epoch 10 (when all 1000 samples have been consumed).
for epoch in range(20):
    for batch in loader:
        (audio, audio_lengths), (tokens, token_lengths) = batch
        pass  # train step
    # report metrics / save checkpoint every mini-epoch
    print(f"mini-epoch {epoch}: done")
```

### Resuming from a checkpoint with `set_mini_epoch`

For sequential (no-shuffle) training on very large data that is only seen once,
`set_mini_epoch(n)` repositions the loader at the exact point where a previous
run left off.  Save the mini-epoch counter in your checkpoint; on resume, call
`set_mini_epoch` before iterating.

```python
# -- checkpoint saved at mini-epoch 34 --
# ckpt = torch.load('checkpoint.pt')
# model.load_state_dict(ckpt['model'])
# optimizer.load_state_dict(ckpt['optimizer'])

loader.set_mini_epoch(34)   # fast-forward to mini-epoch 34

for epoch in range(34, total_mini_epochs):
    for batch in loader:
        pass  # train step
    # save checkpoint with current mini-epoch
```

> **Note:** With `shuffle=False` (sequential data), `set_mini_epoch` produces
> an order **identical** to an uninterrupted run.  With `shuffle=True`, each
> internal `reset()` generates a new random permutation, so exact
> reproducibility requires external seed control.

## VVTKDataLoader Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `VVTKDataset` | — | Dataset opened in read mode |
| `batch_size` | `int` | `32` | Samples per batch |
| `num_workers` | `int` | `4` | C++ prefetch threads |
| `ring_size` | `int` | `4` | Pre-allocated batch buffers |
| `shapes` | `list[tuple]` | — | Max shape per item (required) |
| `dtypes` | `list[torch.dtype]` | — | Target dtype per item (required) |
| `padding_values` | `list[float]` | `[0.0, …]` | Fill value per item |
| `shuffle` | `bool` | `False` | Shuffle each epoch |
| `nb_samples_per_epoch` | `int` | `None` | If set, each iteration yields at most this many samples (rounded to full batches). The full dataset is consumed across consecutive mini-epochs and only reshuffled once all samples have been seen. |
| `drop_last` | `bool` | `False` | Drop incomplete final batch |

> **Note:** The C++ loader always emits full batches. If the dataset size is not
> divisible by `batch_size`, the final batch is zero-padded (lengths = 0).
> Use `drop_last=True` to skip it.

## Compression Modes

| Mode | Use case | C++ decode |
|------|----------|------------|
| `none` | Fixed-size tensors | zero-copy mmap |
| `flac` | Audio waveforms | dr_flac (SSE4.1/AVX2) |
| `zstd` | General-purpose | bundled zstddeclib |

Modes are per-tensor: `compression=['flac', 'none']` compresses audio but not tokens.

## Architecture

```
vvtk_dataset/
├── base.py          # Shard writer, mmap reader, header serialization
├── datasets.py      # VVTKDataset: add() / __getitem__() with compression
├── loader.py        # Python wrapper around C++ ring-buffered loader
csrc/
├── vvtk_lib.cpp     # C++ core: mmap reader + multi-threaded batch loader
├── zstddeclib.c     # Zstandard decoder (BSD option used from upstream dual license)
├── dr_flac.h        # Single-header FLAC decoder (Unlicense/MIT-0)
└── dr_flac_impl.c   # dr_flac build unit
```

## Examples

| Example | Description |
|---------|-------------|
| `example11` | CIFAR-100 CNN — `VVTKDataset` + PyTorch `DataLoader` |
| `example12` | CIFAR-100 CNN — `VVTKDataLoader` (C++) |
| `example21` | Synthetic image benchmark — PyTorch DL vs VVTK + PyTorch DL |
| `example22` | Synthetic image benchmark — PyTorch DL vs `VVTKDataLoader` |
| `example23` | Same as 22 with shuffle |
| `example31` | Synthetic audio + tokens — PyTorch DL vs `VVTKDataLoader` |
| `example41` | CIFAR-100 compression benchmark — uncompressed vs zstd-compressed VVTK shards |
| `example51` | Audio mini-epochs — `nb_samples_per_epoch` with PyTorch DL vs `VVTKDataLoader` |

## Tests

```bash
python tests.py                          # quick: fixed + variable tensors
python -m pytest tests/ -q               # full suite (77 tests)
```

## License

This project is licensed under the **BSD 3-Clause License**. See [LICENSE](LICENSE).

Bundled third-party components in [csrc/](csrc/) remain under their own licenses:

- `zstddeclib.c` — upstream dual-licensed; this project uses the BSD-style option. See [csrc/LICENSE_zstd](csrc/LICENSE_zstd).
- `dr_flac.h` / `dr_flac_impl.c` — upstream MIT-0 / public-domain choice. See [csrc/LICENSE_dr_flac](csrc/LICENSE_dr_flac).

See [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) for a summary.
