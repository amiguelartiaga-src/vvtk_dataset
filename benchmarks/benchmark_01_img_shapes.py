#!/usr/bin/env python3
"""
Benchmark 01 — Fixed-shape image tensors.

Three systems benchmarked, each with its own freshly-generated data so
no system benefits from another's OS page-cache residue.  1 epoch each.

  A. Raw .npy files  + PyTorch Dataset/DataLoader  (baseline)
  B. VVTKDataset     + PyTorch DataLoader
  C. VVTKDataset     + VVTKDataLoader (C++)

Speedup is measured relative to System A.
"""

import os
import sys
import shutil
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vvtk_dataset import VVTKDataset, VVTKDataLoader

# ── Config ───────────────────────────────────────────────────────────────────

N             = 1000
IMG_SHAPE     = (3, 112, 112)
LBL_SHAPE     = (1,)
BATCH_SIZE    = 64
NUM_WORKERS   = 4
NUM_SHARDS    = 8
DATA_DIR      = os.path.join(os.path.dirname(__file__), 'data')

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

def _make_sample(rng, i):
    img = rng.integers(0, 256, size=IMG_SHAPE, dtype=np.uint8)
    lbl = np.array([i % 100], dtype=np.int64)
    return img, lbl


def generate_npy(folder):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(N):
        img, lbl = _make_sample(rng, i)
        np.save(os.path.join(folder, f'{i}_0.npy'), img)
        np.save(os.path.join(folder, f'{i}_1.npy'), lbl)


def generate_vvtk(vvtk_path):
    os.makedirs(os.path.dirname(vvtk_path), exist_ok=True)
    rng = np.random.default_rng(42)
    with VVTKDataset(vvtk_path, mode='wb', num_shards=NUM_SHARDS,
                     compression=['none', 'none']) as ds:
        for i in range(N):
            img, lbl = _make_sample(rng, i)
            ds.add(i, img, lbl)

# ── Helpers ──────────────────────────────────────────────────────────────────

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
    print(f'    1 epoch: {elapsed:6.2f}s  '
          f'({batches:>4} batches, {rate:8.0f} samples/s)')
    return rate


def cleanup():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR, ignore_errors=True)
    print('\n[Cleanup] Removed benchmark data.')

# ── System A: .npy + PyTorch Dataset/DataLoader (baseline) ──────────────────

def bench_npy_torch():
    bench_header('A. Raw .npy + PyTorch Dataset/DataLoader  (baseline)')
    folder = os.path.join(DATA_DIR, 'bench01_a')

    print(f'    Generating {N} .npy files ...')
    t0 = time.time()
    generate_npy(folder)
    print(f'    Generated in {time.time() - t0:.2f}s')

    ds = NpyImageDataset(folder, N)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True)
    return run_one_epoch(loader, lambda b: b)

# ── System B: VVTKDataset + PyTorch DataLoader ──────────────────────────────

def bench_vvtk_torch():
    bench_header('B. VVTKDataset + PyTorch DataLoader')
    vvtk_path = os.path.join(DATA_DIR, 'bench01_b', 'train')

    print(f'    Generating VVTK shards ...')
    t0 = time.time()
    generate_vvtk(vvtk_path)
    print(f'    Generated in {time.time() - t0:.2f}s')

    ds = VVTKDataset(vvtk_path, mode='rb', compression=['none', 'none'])
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True)
    rate = run_one_epoch(loader, lambda b: b)
    ds.close()
    return rate

# ── System C: VVTKDataset + VVTKDataLoader (C++) ────────────────────────────

def bench_vvtk_cpp():
    bench_header('C. VVTKDataset + VVTKDataLoader (C++)')
    vvtk_path = os.path.join(DATA_DIR, 'bench01_c', 'train')

    print(f'    Generating VVTK shards ...')
    t0 = time.time()
    generate_vvtk(vvtk_path)
    print(f'    Generated in {time.time() - t0:.2f}s')

    ds = VVTKDataset(vvtk_path, mode='rb', compression=['none', 'none'])
    loader = VVTKDataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        ring_size=4,
        shapes=[IMG_SHAPE, LBL_SHAPE],
        dtypes=[torch.uint8, torch.int64],
        shuffle=True,
    )
    rate = run_one_epoch(loader, lambda b: b)
    ds.close()
    return rate

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  Benchmark 01 — Fixed-shape images (3×112×112 uint8)')
    print(f'  N={N}  batch={BATCH_SIZE}  workers={NUM_WORKERS}  '
          f'shards={NUM_SHARDS}')
    print(f'  Fresh data per system · 1 epoch each')
    print('=' * 60 + '\n')

    results = {}
    results['npy'] = bench_npy_torch()
    print()
    results['vvtk_torch'] = bench_vvtk_torch()
    print()
    results['vvtk_cpp'] = bench_vvtk_cpp()

    baseline = results['npy']
    print(f'\n{"=" * 60}')
    print('  Summary')
    print(f'{"=" * 60}')
    r_bt = results['vvtk_torch'] / baseline if baseline > 0 else 0
    r_bc = results['vvtk_cpp']   / baseline if baseline > 0 else 0
    print(f'  A. .npy + Torch DL (baseline): {baseline:>10.0f} samples/s')
    print(f'  B. VVTK + Torch DL           : {results["vvtk_torch"]:>10.0f} samples/s  '
          f'({r_bt:>6.2f}x)')
    print(f'  C. VVTK + VVTK C++ DL        : {results["vvtk_cpp"]:>10.0f} samples/s  '
          f'({r_bc:>6.2f}x)')
    print(f'{"=" * 60}\n')

    cleanup()


if __name__ == '__main__':
    main()
