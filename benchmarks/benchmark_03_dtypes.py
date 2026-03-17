#!/usr/bin/env python3
"""
Benchmark 03 — Multiple dtypes with random data tensors.

Three systems benchmarked per dtype, each with its own freshly-generated
data so no system benefits from another's OS page-cache residue.
1 epoch each.

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
BATCH_SIZE    = 64
NUM_WORKERS   = 4
NUM_SHARDS    = 8
DATA_DIR      = os.path.join(os.path.dirname(__file__), 'data')

# (label, np_dtype, torch_dtype, shape, value_range)
DTYPE_CONFIGS = [
    ('uint8',   np.uint8,   torch.uint8,   (3, 32, 32), (0, 255)),
    ('int16',   np.int16,   torch.int16,   (1024,),     (-1000, 1000)),
    ('int32',   np.int32,   torch.int32,   (512,),      (-100000, 100000)),
    ('int64',   np.int64,   torch.int64,   (256,),      (0, 1000000)),
    ('float16', np.float16, torch.float16, (2048,),     (-1.0, 1.0)),
    ('float32', np.float32, torch.float32, (3, 64, 64), (-1.0, 1.0)),
    ('float64', np.float64, torch.float64, (1024,),     (-1.0, 1.0)),
]

# ── Baseline Dataset ─────────────────────────────────────────────────────────

class NpyDataset(Dataset):
    """Standard file-per-sample dataset (typical PyTorch baseline)."""

    def __init__(self, folder, n):
        self.folder = folder
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.from_numpy(np.load(os.path.join(self.folder, f'{idx}.npy')))

# ── Generators ───────────────────────────────────────────────────────────────

def _make_data(rng, np_dtype, shape, value_range):
    lo, hi = value_range
    if np.issubdtype(np_dtype, np.integer):
        return rng.integers(lo, hi + 1, size=shape, dtype=np_dtype)
    return rng.uniform(lo, hi, size=shape).astype(np_dtype)


def generate_npy(folder, np_dtype, shape, value_range):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(N):
        data = _make_data(rng, np_dtype, shape, value_range)
        np.save(os.path.join(folder, f'{i}.npy'), data)


def generate_vvtk(vvtk_path, np_dtype, shape, value_range):
    os.makedirs(os.path.dirname(vvtk_path), exist_ok=True)
    rng = np.random.default_rng(42)
    with VVTKDataset(vvtk_path, mode='wb', num_shards=NUM_SHARDS,
                     compression=['none']) as ds:
        for i in range(N):
            data = _make_data(rng, np_dtype, shape, value_range)
            ds.add(i, data)

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
    print(f'      1 epoch: {elapsed:6.2f}s  '
          f'({batches:>4} batches, {rate:8.0f} samples/s)')
    return rate


def bench_one_dtype(label, np_dtype, torch_dtype, shape, value_range):
    """Benchmark a single dtype: 3 systems, fresh data each, 1 epoch."""
    print(f'\n  ▸ {label}  shape={shape}  range={value_range}')

    # ── System A: .npy baseline ──
    print(f'    A. .npy + Torch DL:')
    folder_a = os.path.join(DATA_DIR, 'bench03', label, 'a')
    ta = time.time()
    generate_npy(folder_a, np_dtype, shape, value_range)
    print(f'      Generated .npy in {time.time() - ta:.2f}s')
    ds_a = NpyDataset(folder_a, N)
    loader_a = DataLoader(ds_a, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    rate_npy = run_one_epoch(loader_a, lambda b: b)

    # ── System B: VVTK + Torch DL ──
    print(f'    B. VVTK + Torch DL:')
    vvtk_b = os.path.join(DATA_DIR, 'bench03', label, 'b', 'data')
    tb = time.time()
    generate_vvtk(vvtk_b, np_dtype, shape, value_range)
    print(f'      Generated VVTK in {time.time() - tb:.2f}s')
    ds_b = VVTKDataset(vvtk_b, mode='rb', compression=['none'])
    loader_b = DataLoader(ds_b, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    rate_vvtk_torch = run_one_epoch(loader_b, lambda b: b)
    ds_b.close()

    # ── System C: VVTK + C++ DL ──
    print(f'    C. VVTK + VVTK C++ DL:')
    vvtk_c = os.path.join(DATA_DIR, 'bench03', label, 'c', 'data')
    tc = time.time()
    generate_vvtk(vvtk_c, np_dtype, shape, value_range)
    print(f'      Generated VVTK in {time.time() - tc:.2f}s')
    ds_c = VVTKDataset(vvtk_c, mode='rb', compression=['none'])
    loader_c = VVTKDataLoader(
        ds_c,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        ring_size=4,
        shapes=[shape],
        dtypes=[torch_dtype],
        shuffle=True,
    )
    rate_vvtk_cpp = run_one_epoch(loader_c, lambda b: b)
    ds_c.close()

    return rate_npy, rate_vvtk_torch, rate_vvtk_cpp


def cleanup():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR, ignore_errors=True)
    print('\n[Cleanup] Removed benchmark data.')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  Benchmark 03 — Dtype comparison (random data tensors)')
    print(f'  N={N}  batch={BATCH_SIZE}  workers={NUM_WORKERS}  '
          f'shards={NUM_SHARDS}')
    print(f'  Fresh data per system · 1 epoch each')
    print('=' * 60)

    results = {}
    for label, np_dt, torch_dt, shape, vrange in DTYPE_CONFIGS:
        r_npy, r_vt, r_vc = bench_one_dtype(label, np_dt, torch_dt, shape, vrange)
        results[label] = (r_npy, r_vt, r_vc)

    # Summary table
    print(f'\n{"=" * 60}')
    print('  Summary  (speedup vs .npy baseline)')
    print(f'{"=" * 60}')
    print(f'  {"Dtype":<10} {"Shape":<16} '
          f'{"A .npy":>10} {"B VVTK+T":>10} {"C VVTK+C":>10} '
          f'{"B/A":>7} {"C/A":>7}')
    print(f'  {"─" * 10} {"─" * 16} '
          f'{"─" * 10} {"─" * 10} {"─" * 10} '
          f'{"─" * 7} {"─" * 7}')

    for label, np_dt, torch_dt, shape, vrange in DTYPE_CONFIGS:
        rn, rvt, rvc = results[label]
        sb = rvt / rn if rn > 0 else 0
        sc = rvc / rn if rn > 0 else 0
        print(f'  {label:<10} {str(shape):<16} '
              f'{rn:>8.0f}/s {rvt:>8.0f}/s {rvc:>8.0f}/s '
              f'{sb:>6.2f}x {sc:>6.2f}x')

    print(f'{"=" * 60}\n')

    cleanup()


if __name__ == '__main__':
    main()
