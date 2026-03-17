#!/usr/bin/env python3
"""
Benchmark 04 — Uncompressed vs zstd-compressed fixed-shape image tensors.

Images are sparse (empty canvas with scattered random points) then smoothed
with two random Gaussian filters, producing compressible content that
exercises zstd meaningfully.

Five systems benchmarked per image shape, each with its own freshly-generated
data so no system benefits from another's OS page-cache residue.
1 epoch each.  Shard sizes reported for none vs zstd.

  A.  Raw .npy files  + PyTorch Dataset/DataLoader   (reference)
  B1. VVTKDataset (none) + PyTorch DataLoader
  B2. VVTKDataset (zstd) + PyTorch DataLoader
  C1. VVTKDataset (none) + VVTKDataLoader (C++)
  C2. VVTKDataset (zstd) + VVTKDataLoader (C++)

Speedup is measured relative to System A.
"""
# remove all warnings to keep output clean (e.g. from PyTorch about pinned memory) or numpy disk warnings about memmap
import warnings
warnings.filterwarnings("ignore")

import glob
import os
import sys
import shutil
import time
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vvtk_dataset import VVTKDataset, VVTKDataLoader

# ── Config ───────────────────────────────────────────────────────────────────

N             = 1000
LBL_SHAPE     = (1,)
BATCH_SIZE    = 64
NUM_WORKERS   = 4
NUM_SHARDS    = 8
ZSTD_LEVEL    = 3
SPARSITY      = 0.02          # fraction of non-zero pixels
DATA_DIR      = os.path.join(os.path.dirname(__file__), 'data')

# (label, shape)
IMG_SHAPE_CONFIGS = [
    ('3x32x32',   (3,  32,  32)),
    ('3x64x64',   (3,  64,  64)),
    ('3x112x112', (3, 112, 112)),
    ('3x224x224', (3, 224, 224)),
    ('1x256x256', (1, 256, 256)),
    ('3x512x512', (3, 512, 512)),
]

# ── Baseline Dataset ─────────────────────────────────────────────────────────

class NpyImageDataset(Dataset):
    """Standard file-per-sample dataset (typical PyTorch baseline)."""

    def __init__(self, folder, n):
        self.folder = folder
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = torch.from_numpy(np.load(os.path.join(self.folder, f'{idx}_0.npy')))
        lbl = torch.from_numpy(np.load(os.path.join(self.folder, f'{idx}_1.npy')))
        return img, lbl

# ── Generators ───────────────────────────────────────────────────────────────

def _gaussian_kernel_2d(rng, h, w):
    """Build a random 2-D Gaussian filter and apply it (per-channel)."""
    # Random kernel size: odd number in [3, max(3, min(h,w)//4)]
    max_k = max(3, min(h, w) // 4)
    k = rng.integers(1, max_k // 2 + 1) * 2 + 1          # odd
    sigma = rng.uniform(0.5, k / 2.0)
    return k, sigma


def _make_sample(rng, img_shape, i):
    """
    Generate a sparse image filtered with two random Gaussian kernels.

    1. Start with a blank (zeros) image.
    2. Scatter sparse random-valued points (SPARSITY fraction).
    3. Apply two successive 2-D Gaussian filters with random kernel sizes
       to produce smooth blobs, then clip & cast back to uint8.
    """
    C, H, W = img_shape
    img = np.zeros(img_shape, dtype=np.float32)

    # Scatter sparse random points
    n_points = max(1, int(H * W * SPARSITY))
    for c in range(C):
        rows = rng.integers(0, H, size=n_points)
        cols = rng.integers(0, W, size=n_points)
        vals = rng.integers(1, 256, size=n_points).astype(np.float32)
        img[c, rows, cols] = vals

    # Apply two Gaussian filters with random kernel sizes
    k1, sigma1 = _gaussian_kernel_2d(rng, H, W)
    k2, sigma2 = _gaussian_kernel_2d(rng, H, W)

    for c in range(C):
        img[c] = gaussian_filter(img[c], sigma=sigma1, truncate=(k1 / (2 * sigma1)))
        img[c] = gaussian_filter(img[c], sigma=sigma2, truncate=(k2 / (2 * sigma2)))

    # Normalize to [0, 255] and cast to uint8
    img_max = img.max()
    if img_max > 0:
        img = img * (255.0 / img_max)
    img = np.clip(img, 0, 255).astype(np.uint8)

    lbl = np.array([i % 100], dtype=np.int64)
    return img, lbl


def generate_npy(folder, img_shape):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(N):
        img, lbl = _make_sample(rng, img_shape, i)
        np.save(os.path.join(folder, f'{i}_0.npy'), img)
        np.save(os.path.join(folder, f'{i}_1.npy'), lbl)


def generate_vvtk_from_npy(vvtk_path, npy_folder, compression_mode):
    """Build VVTK shards by reading the already-generated .npy files."""
    os.makedirs(os.path.dirname(vvtk_path), exist_ok=True)
    comp = [compression_mode, 'none']
    comp_args = [{'level': ZSTD_LEVEL} if compression_mode == 'zstd' else {}, {}]
    with VVTKDataset(vvtk_path, mode='wb', num_shards=NUM_SHARDS,
                     compression=comp,
                     compression_args=comp_args) as ds:
        for i in range(N):
            img = np.load(os.path.join(npy_folder, f'{i}_0.npy'))
            lbl = np.load(os.path.join(npy_folder, f'{i}_1.npy'))
            ds.add(i, img, lbl)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_size(nbytes):
    """Human-readable byte size."""
    if nbytes < 1024:
        return f'{nbytes} B'
    elif nbytes < 1024 ** 2:
        return f'{nbytes / 1024:.1f} KB'
    elif nbytes < 1024 ** 3:
        return f'{nbytes / 1024 ** 2:.1f} MB'
    return f'{nbytes / 1024 ** 3:.2f} GB'


def measure_vvtk_size(vvtk_path):
    """Total size of all .vvtk shard files for a given path prefix."""
    total = 0
    for f in glob.glob(f'{vvtk_path}.*.vvtk'):
        total += os.path.getsize(f)
    return total


def measure_npy_size(folder):
    """Total size of all .npy files in a folder."""
    total = 0
    for f in glob.glob(os.path.join(folder, '*.npy')):
        total += os.path.getsize(f)
    return total


def bench_header(name):
    print(f'{"─" * 60}')
    print(f'  {name}')
    print(f'{"─" * 60}')


def run_one_epoch(loader, unpack_fn):
    t0 = time.time()
    batches = 0
    for batch in loader:
        _ = unpack_fn(batch)
        batches += 1
    elapsed = time.time() - t0
    rate = N / elapsed
    print(f'      1 epoch: {elapsed:6.2f}s  '
          f'({batches:>4} batches, {rate:8.0f} samples/s)')
    return rate


def bench_one_shape(label, img_shape):
    """Benchmark a single image shape: 5 systems, fresh data each, 1 epoch."""
    print(f'\n  ▸ {label}  shape={img_shape}')

    sizes = {}

    # ── System A: .npy + Torch DL (reference) ──
    print(f'    A.  .npy + Torch DL (reference):')
    folder_a = os.path.join(DATA_DIR, 'bench04', label, 'a')
    ta = time.time()
    generate_npy(folder_a, img_shape)
    print(f'      Generated .npy in {time.time() - ta:.2f}s')
    sizes['npy'] = measure_npy_size(folder_a)
    print(f'      Size: {_fmt_size(sizes["npy"])}')
    ds_a = NpyImageDataset(folder_a, N)
    loader_a = DataLoader(ds_a, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    rate_a = run_one_epoch(loader_a, lambda b: b)

    # ── System B1: VVTK (none) + Torch DL ──
    print(f'    B1. VVTK (none) + Torch DL:')
    vvtk_b1 = os.path.join(DATA_DIR, 'bench04', label, 'b1', 'data')
    tb1 = time.time()
    generate_vvtk_from_npy(vvtk_b1, folder_a, 'none')
    print(f'      Generated VVTK (none) in {time.time() - tb1:.2f}s')
    sizes['vvtk_none'] = measure_vvtk_size(vvtk_b1)
    print(f'      Size: {_fmt_size(sizes["vvtk_none"])}')
    ds_b1 = VVTKDataset(vvtk_b1, mode='rb', compression=['none', 'none'])
    loader_b1 = DataLoader(ds_b1, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)
    rate_b1 = run_one_epoch(loader_b1, lambda b: b)
    ds_b1.close()

    # ── System B2: VVTK (zstd) + Torch DL ──
    print(f'    B2. VVTK (zstd) + Torch DL:')
    vvtk_b2 = os.path.join(DATA_DIR, 'bench04', label, 'b2', 'data')
    tb2 = time.time()
    generate_vvtk_from_npy(vvtk_b2, folder_a, 'zstd')
    print(f'      Generated VVTK (zstd) in {time.time() - tb2:.2f}s')
    sizes['vvtk_zstd'] = measure_vvtk_size(vvtk_b2)
    print(f'      Size: {_fmt_size(sizes["vvtk_zstd"])}')
    ds_b2 = VVTKDataset(vvtk_b2, mode='rb', compression=['zstd', 'none'])
    loader_b2 = DataLoader(ds_b2, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)
    rate_b2 = run_one_epoch(loader_b2, lambda b: b)
    ds_b2.close()

    # ── System C1: VVTK (none) + VVTK C++ DL ──
    print(f'    C1. VVTK (none) + VVTK C++ DL:')
    vvtk_c1 = os.path.join(DATA_DIR, 'bench04', label, 'c1', 'data')
    tc1 = time.time()
    generate_vvtk_from_npy(vvtk_c1, folder_a, 'none')
    print(f'      Generated VVTK (none) in {time.time() - tc1:.2f}s')
    ds_c1 = VVTKDataset(vvtk_c1, mode='rb', compression=['none', 'none'])
    loader_c1 = VVTKDataLoader(
        ds_c1,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        ring_size=4,
        shapes=[img_shape, LBL_SHAPE],
        dtypes=[torch.uint8, torch.int64],
        shuffle=True,
    )
    rate_c1 = run_one_epoch(loader_c1, lambda b: b)
    ds_c1.close()

    # ── System C2: VVTK (zstd) + VVTK C++ DL ──
    print(f'    C2. VVTK (zstd) + VVTK C++ DL:')
    vvtk_c2 = os.path.join(DATA_DIR, 'bench04', label, 'c2', 'data')
    tc2 = time.time()
    generate_vvtk_from_npy(vvtk_c2, folder_a, 'zstd')
    print(f'      Generated VVTK (zstd) in {time.time() - tc2:.2f}s')
    ds_c2 = VVTKDataset(vvtk_c2, mode='rb', compression=['zstd', 'none'])
    loader_c2 = VVTKDataLoader(
        ds_c2,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        ring_size=4,
        shapes=[img_shape, LBL_SHAPE],
        dtypes=[torch.uint8, torch.int64],
        shuffle=True,
    )
    rate_c2 = run_one_epoch(loader_c2, lambda b: b)
    ds_c2.close()

    return rate_a, rate_b1, rate_b2, rate_c1, rate_c2, sizes


def cleanup():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR, ignore_errors=True)
    print('\n[Cleanup] Removed benchmark data.')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    W = 120
    print('=' * W)
    print('  Benchmark 04 — Uncompressed vs zstd compressed images, varying shapes')
    print(f'  N={N}  batch={BATCH_SIZE}  workers={NUM_WORKERS}  '
          f'shards={NUM_SHARDS}  zstd_level={ZSTD_LEVEL}')
    print(f'  Fresh data per system · 1 epoch each')
    print('=' * W)

    results = {}
    all_sizes = {}
    for label, img_shape in IMG_SHAPE_CONFIGS:
        r_a, r_b1, r_b2, r_c1, r_c2, sz = bench_one_shape(label, img_shape)
        results[label] = (r_a, r_b1, r_b2, r_c1, r_c2)
        all_sizes[label] = sz

    # ── Throughput summary ──
    print(f'\n{"=" * W}')
    print('  Throughput Summary  (speedup vs A = .npy + Torch)')
    print(f'{"=" * W}')
    print(f'  {"Shape":<14} '
          f'{"A .npy+T":>10} '
          f'{"B1 none+T":>11} '
          f'{"B2 zstd+T":>11} '
          f'{"C1 none+C":>11} '
          f'{"C2 zstd+C":>11} '
          f'{"B1/A":>7} {"B2/A":>7} {"C1/A":>7} {"C2/A":>7}')
    print(f'  {"─" * 14} '
          f'{"─" * 10} '
          f'{"─" * 11} '
          f'{"─" * 11} '
          f'{"─" * 11} '
          f'{"─" * 11} '
          f'{"─" * 7} {"─" * 7} {"─" * 7} {"─" * 7}')

    for label, _ in IMG_SHAPE_CONFIGS:
        ra, rb1, rb2, rc1, rc2 = results[label]
        s_b1 = rb1 / ra if ra > 0 else 0
        s_b2 = rb2 / ra if ra > 0 else 0
        s_c1 = rc1 / ra if ra > 0 else 0
        s_c2 = rc2 / ra if ra > 0 else 0
        print(f'  {label:<14} '
              f'{ra:>8.0f}/s '
              f'{rb1:>9.0f}/s '
              f'{rb2:>9.0f}/s '
              f'{rc1:>9.0f}/s '
              f'{rc2:>9.0f}/s '
              f'{s_b1:>6.2f}x {s_b2:>6.2f}x {s_c1:>6.2f}x {s_c2:>6.2f}x')

    # ── Storage summary ──
    print(f'\n{"=" * W}')
    print('  Storage Summary  (shard sizes)')
    print(f'{"=" * W}')
    print(f'  {"Shape":<14} '
          f'{"npy total":>12} '
          f'{"VVTK none":>12} '
          f'{"VVTK zstd":>12} '
          f'{"zstd/none":>10} '
          f'{"zstd/npy":>10}')
    print(f'  {"─" * 14} '
          f'{"─" * 12} '
          f'{"─" * 12} '
          f'{"─" * 12} '
          f'{"─" * 10} '
          f'{"─" * 10}')

    for label, _ in IMG_SHAPE_CONFIGS:
        sz = all_sizes[label]
        sz_npy  = sz['npy']
        sz_none = sz['vvtk_none']
        sz_zstd = sz['vvtk_zstd']
        ratio_zn = sz_zstd / sz_none if sz_none > 0 else 0
        ratio_zp = sz_zstd / sz_npy  if sz_npy  > 0 else 0
        print(f'  {label:<14} '
              f'{_fmt_size(sz_npy):>12} '
              f'{_fmt_size(sz_none):>12} '
              f'{_fmt_size(sz_zstd):>12} '
              f'{ratio_zn:>9.2f}x '
              f'{ratio_zp:>9.2f}x')

    print(f'{"=" * W}\n')

    cleanup()


if __name__ == '__main__':
    main()
