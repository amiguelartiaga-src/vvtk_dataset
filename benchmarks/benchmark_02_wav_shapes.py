#!/usr/bin/env python3
"""
Benchmark 02 — Variable-length audio-like tensors with padding.

Three systems benchmarked, each with its own freshly-generated data so
no system benefits from another's OS page-cache residue.  1 epoch each.

  A. Raw .npy files  + PyTorch Dataset/DataLoader  (baseline, with padding)
  B. VVTKDataset     + PyTorch DataLoader           (with padding)
  C. VVTKDataset     + VVTKDataLoader (C++)          (auto-padding)

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
SR            = 16_000
MIN_AUDIO     = 1000
MAX_AUDIO     = 32000
MIN_TOK       = 10
MAX_TOK       = 400
MAX_AUDIO_PAD = MAX_AUDIO       # pad/truncate target
MAX_TOK_PAD   = MAX_TOK
BATCH_SIZE    = 32
NUM_WORKERS   = 4
NUM_SHARDS    = 8
DATA_DIR      = os.path.join(os.path.dirname(__file__), 'data')

# ── Baseline Dataset ─────────────────────────────────────────────────────────

class NpyAudioDataset(Dataset):
    """Standard file-per-sample dataset with padding (typical baseline)."""

    def __init__(self, folder, n, max_audio, max_tok):
        self.folder = folder
        self.n = n
        self.max_audio = max_audio
        self.max_tok = max_tok

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        audio = np.load(os.path.join(self.folder, f'{idx}_0.npy'))
        tokens = np.load(os.path.join(self.folder, f'{idx}_1.npy'))

        audio_len = min(len(audio), self.max_audio)
        tok_len = min(len(tokens), self.max_tok)

        padded_audio = np.zeros(self.max_audio, dtype=np.float32)
        padded_audio[:audio_len] = audio[:audio_len]

        padded_tokens = np.zeros(self.max_tok, dtype=np.int16)
        padded_tokens[:tok_len] = tokens[:tok_len]

        return ((torch.from_numpy(padded_audio),
                 torch.tensor(audio_len, dtype=torch.int64)),
                (torch.from_numpy(padded_tokens),
                 torch.tensor(tok_len, dtype=torch.int64)))

# ── Generators ───────────────────────────────────────────────────────────────

def _get_lengths(i):
    rs = np.random.RandomState(seed=i)
    return rs.randint(MIN_AUDIO, MAX_AUDIO + 1), rs.randint(MIN_TOK, MAX_TOK + 1)


def _make_sample(rng, i):
    audio_len, tok_len = _get_lengths(i)
    freq = rng.uniform(200, 4000)
    t = np.arange(audio_len, dtype=np.float32) / SR
    waveform = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    tokens = rng.integers(0, 30000, size=tok_len, dtype=np.int16)
    return waveform, tokens


def generate_npy(folder):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(N):
        waveform, tokens = _make_sample(rng, i)
        np.save(os.path.join(folder, f'{i}_0.npy'), waveform)
        np.save(os.path.join(folder, f'{i}_1.npy'), tokens)


def generate_vvtk(vvtk_path):
    os.makedirs(os.path.dirname(vvtk_path), exist_ok=True)
    rng = np.random.default_rng(42)
    with VVTKDataset(vvtk_path, mode='wb', num_shards=NUM_SHARDS,
                     compression=['none', 'none']) as ds:
        for i in range(N):
            waveform, tokens = _make_sample(rng, i)
            ds.add(i, waveform, tokens)

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
    folder = os.path.join(DATA_DIR, 'bench02_a')

    print(f'    Generating {N} .npy files ...')
    t0 = time.time()
    generate_npy(folder)
    print(f'    Generated in {time.time() - t0:.2f}s')

    ds = NpyAudioDataset(folder, N, MAX_AUDIO_PAD, MAX_TOK_PAD)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True)
    return run_one_epoch(loader, lambda b: b)

# ── System B: VVTKDataset (padded) + PyTorch DataLoader ─────────────────────

def bench_vvtk_torch():
    bench_header('B. VVTKDataset (padded) + PyTorch DataLoader')
    vvtk_path = os.path.join(DATA_DIR, 'bench02_b', 'train')

    print(f'    Generating VVTK shards ...')
    t0 = time.time()
    generate_vvtk(vvtk_path)
    print(f'    Generated in {time.time() - t0:.2f}s')

    ds = VVTKDataset(vvtk_path, mode='rb', compression=['none', 'none'],
                     fixed_shapes=[(MAX_AUDIO_PAD,), (MAX_TOK_PAD,)],
                     padding_values=[0.0, 0])
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True)
    rate = run_one_epoch(loader, lambda b: b)
    ds.close()
    return rate

# ── System C: VVTKDataset + VVTKDataLoader (C++) ────────────────────────────

def bench_vvtk_cpp():
    bench_header('C. VVTKDataset + VVTKDataLoader (C++ auto-padding)')
    vvtk_path = os.path.join(DATA_DIR, 'bench02_c', 'train')

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
        shapes=[(MAX_AUDIO_PAD,), (MAX_TOK_PAD,)],
        dtypes=[torch.float32, torch.int16],
        padding_values=[0.0, 0],
        shuffle=True,
    )
    rate = run_one_epoch(loader, lambda b: b)
    ds.close()
    return rate

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  Benchmark 02 — Variable-length audio + tokens (padding)')
    print(f'  N={N}  batch={BATCH_SIZE}  workers={NUM_WORKERS}  '
          f'shards={NUM_SHARDS}')
    print(f'  Audio: {MIN_AUDIO}–{MAX_AUDIO} float32  '
          f'Tokens: {MIN_TOK}–{MAX_TOK} int16')
    print(f'  Pad to: audio={MAX_AUDIO_PAD}  tokens={MAX_TOK_PAD}')
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
