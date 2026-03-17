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

### After upgrading PyTorch

The C++ extension must be rebuilt whenever PyTorch is upgraded:

```bash
python fix.py
```

This removes stale `.so` files, rebuilds, and verifies the import.

### `GLIBCXX_X.Y.Z not found`

Conda may ship an older `libstdc++` than required. Fix:

```bash
conda install -c conda-forge libstdcxx-ng>=12
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
