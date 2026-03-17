#!/usr/bin/env python3
"""
Benchmark 05 — Uncompressed vs FLAC-compressed variable-length sinusoid audio.

Five systems benchmarked per audio-duration config, each with its own
freshly-generated data so no system benefits from another's OS page-cache
residue.  1 epoch each.  Shard sizes reported for none vs flac.

  A.  Raw .wav files  + PyTorch Dataset/DataLoader  (baseline)
  B1. VVTKDataset (none) + PyTorch DataLoader
  B2. VVTKDataset (flac) + PyTorch DataLoader
  C1. VVTKDataset (none) + VVTKDataLoader (C++)
  C2. VVTKDataset (flac) + VVTKDataLoader (C++ dr_flac)

Speedup is measured relative to System A.
"""
import warnings
warnings.filterwarnings("ignore")

import glob
import os
import sys
import shutil
import time
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vvtk_dataset import VVTKDataset, VVTKDataLoader

# ── Config ───────────────────────────────────────────────────────────────────

N             = 500
SR            = 16_000
MIN_TOK       = 10
MAX_TOK       = 200
BATCH_SIZE    = 32
NUM_WORKERS   = 4
NUM_SHARDS    = 8
DATA_DIR      = os.path.join(os.path.dirname(__file__), 'data')

# (label, min_audio_samples, max_audio_samples)
AUDIO_CONFIGS = [
    ('0.25-1s',   4_000,  16_000),
    ('1-5s',     16_000,  80_000),
    ('5-15s',    80_000, 240_000),
    ('15-30s',  240_000, 480_000),
]

# ── Baseline Dataset ─────────────────────────────────────────────────────────

class WavTokenDataset(Dataset):
    """Standard file-per-sample dataset with padding (typical baseline)."""

    def __init__(self, folder, n, max_audio, max_tok):
        self.folder = folder
        self.n = n
        self.max_audio = max_audio
        self.max_tok = max_tok

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        audio, _ = sf.read(os.path.join(self.folder, f'{idx}.wav'), dtype='float32')
        tokens = np.load(os.path.join(self.folder, f'{idx}_tok.npy'))

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

def _get_lengths(i, min_audio, max_audio):
    rs = np.random.RandomState(seed=i)
    return rs.randint(min_audio, max_audio + 1), rs.randint(MIN_TOK, MAX_TOK + 1)


def _make_sample(rng, i, min_audio, max_audio):
    """Generate a sinusoid waveform + random token sequence."""
    audio_len, tok_len = _get_lengths(i, min_audio, max_audio)
    freq = rng.uniform(200, 4000)
    t = np.arange(audio_len, dtype=np.float32) / SR
    waveform = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    tokens = rng.integers(0, 30000, size=tok_len, dtype=np.int16)
    return waveform, tokens


def generate_wav_files(folder, min_audio, max_audio):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(N):
        waveform, tokens = _make_sample(rng, i, min_audio, max_audio)
        sf.write(os.path.join(folder, f'{i}.wav'), waveform, SR)
        np.save(os.path.join(folder, f'{i}_tok.npy'), tokens)


def generate_vvtk(vvtk_path, wav_folder, compression_mode):
    """Build VVTK shards by reading the already-generated .wav + .npy files."""
    os.makedirs(os.path.dirname(vvtk_path), exist_ok=True)
    comp = [compression_mode, 'none']
    comp_args = [{'sample_rate': SR} if compression_mode == 'flac' else {}, {}]
    with VVTKDataset(vvtk_path, mode='wb', num_shards=NUM_SHARDS,
                     compression=comp, compression_args=comp_args) as ds:
        for i in range(N):
            audio, _ = sf.read(os.path.join(wav_folder, f'{i}.wav'), dtype='float32')
            tokens = np.load(os.path.join(wav_folder, f'{i}_tok.npy'))
            ds.add(i, audio, tokens)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_size(nbytes):
    if nbytes < 1024:
        return f'{nbytes} B'
    elif nbytes < 1024 ** 2:
        return f'{nbytes / 1024:.1f} KB'
    elif nbytes < 1024 ** 3:
        return f'{nbytes / 1024 ** 2:.1f} MB'
    return f'{nbytes / 1024 ** 3:.2f} GB'


def measure_vvtk_size(vvtk_path):
    total = 0
    for f in glob.glob(f'{vvtk_path}.*.vvtk'):
        total += os.path.getsize(f)
    return total


def measure_wav_size(folder):
    total = 0
    for f in glob.glob(os.path.join(folder, '*.wav')):
        total += os.path.getsize(f)
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
    return elapsed, rate


def bench_one_config(label, min_audio, max_audio):
    """Benchmark a single audio-duration config: 5 systems, fresh data each."""
    max_pad = max_audio
    print(f'\n  ▸ {label}  audio={min_audio}–{max_audio} samples  '
          f'({min_audio/SR:.2f}–{max_audio/SR:.1f}s)')

    sizes = {}

    # ── Generate .wav baseline data (shared source for all systems) ──
    wav_folder = os.path.join(DATA_DIR, 'bench05', label, 'wav')
    print(f'    Generating {N} .wav + .npy files ...')
    t0 = time.time()
    generate_wav_files(wav_folder, min_audio, max_audio)
    print(f'    Generated in {time.time() - t0:.2f}s')
    sizes['wav'] = measure_wav_size(wav_folder)

    # ── System A: .wav + Torch DL (baseline) ──
    print(f'    A.  .wav + Torch DL (baseline):')
    print(f'      Size: {_fmt_size(sizes["wav"])}')
    ds_a = WavTokenDataset(wav_folder, N, max_pad, MAX_TOK)
    loader_a = DataLoader(ds_a, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    time_a, rate_a = run_one_epoch(loader_a, lambda b: b)

    # ── System B1: VVTK (none) + Torch DL ──
    print(f'    B1. VVTK (none) + Torch DL:')
    vvtk_b1 = os.path.join(DATA_DIR, 'bench05', label, 'b1', 'data')
    tb = time.time()
    generate_vvtk(vvtk_b1, wav_folder, 'none')
    print(f'      Generated VVTK (none) in {time.time() - tb:.2f}s')
    sizes['vvtk_none'] = measure_vvtk_size(vvtk_b1)
    print(f'      Size: {_fmt_size(sizes["vvtk_none"])}')
    ds_b1 = VVTKDataset(vvtk_b1, mode='rb', compression=['none', 'none'],
                        fixed_shapes=[(max_pad,), (MAX_TOK,)],
                        padding_values=[0.0, 0])
    loader_b1 = DataLoader(ds_b1, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)
    time_b1, rate_b1 = run_one_epoch(loader_b1, lambda b: b)
    ds_b1.close()

    # ── System B2: VVTK (flac) + Torch DL ──
    print(f'    B2. VVTK (flac) + Torch DL:')
    vvtk_b2 = os.path.join(DATA_DIR, 'bench05', label, 'b2', 'data')
    tb = time.time()
    generate_vvtk(vvtk_b2, wav_folder, 'flac')
    print(f'      Generated VVTK (flac) in {time.time() - tb:.2f}s')
    sizes['vvtk_flac'] = measure_vvtk_size(vvtk_b2)
    print(f'      Size: {_fmt_size(sizes["vvtk_flac"])}')
    ds_b2 = VVTKDataset(vvtk_b2, mode='rb',
                        compression=['flac', 'none'],
                        compression_args=[{'sample_rate': SR}, {}],
                        fixed_shapes=[(max_pad,), (MAX_TOK,)],
                        padding_values=[0.0, 0])
    loader_b2 = DataLoader(ds_b2, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True)
    time_b2, rate_b2 = run_one_epoch(loader_b2, lambda b: b)
    ds_b2.close()

    # ── System C1: VVTK (none) + VVTK C++ DL ──
    print(f'    C1. VVTK (none) + VVTK C++ DL:')
    vvtk_c1 = os.path.join(DATA_DIR, 'bench05', label, 'c1', 'data')
    tc = time.time()
    generate_vvtk(vvtk_c1, wav_folder, 'none')
    print(f'      Generated VVTK (none) in {time.time() - tc:.2f}s')
    ds_c1 = VVTKDataset(vvtk_c1, mode='rb', compression=['none', 'none'])
    loader_c1 = VVTKDataLoader(
        ds_c1,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        ring_size=4,
        shapes=[(max_pad,), (MAX_TOK,)],
        dtypes=[torch.float32, torch.int16],
        padding_values=[0.0, 0],
        shuffle=True,
    )
    time_c1, rate_c1 = run_one_epoch(loader_c1, lambda b: b)
    ds_c1.close()

    # ── System C2: VVTK (flac) + VVTK C++ DL (dr_flac) ──
    print(f'    C2. VVTK (flac) + VVTK C++ DL (dr_flac):')
    vvtk_c2 = os.path.join(DATA_DIR, 'bench05', label, 'c2', 'data')
    tc = time.time()
    generate_vvtk(vvtk_c2, wav_folder, 'flac')
    print(f'      Generated VVTK (flac) in {time.time() - tc:.2f}s')
    ds_c2 = VVTKDataset(vvtk_c2, mode='rb',
                        compression=['flac', 'none'],
                        compression_args=[{'sample_rate': SR}, {}])
    loader_c2 = VVTKDataLoader(
        ds_c2,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        ring_size=4,
        shapes=[(max_pad,), (MAX_TOK,)],
        dtypes=[torch.float32, torch.int16],
        padding_values=[0.0, 0],
        shuffle=True,
    )
    time_c2, rate_c2 = run_one_epoch(loader_c2, lambda b: b)
    ds_c2.close()

    times  = (time_a, time_b1, time_b2, time_c1, time_c2)
    rates  = (rate_a, rate_b1, rate_b2, rate_c1, rate_c2)
    return rates, times, sizes


def cleanup():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR, ignore_errors=True)
    print('\n[Cleanup] Removed benchmark data.')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    W = 120
    print('=' * W)
    print('  Benchmark 05 — Uncompressed vs FLAC-compressed sinusoid audio + tokens')
    print(f'  N={N}  batch={BATCH_SIZE}  workers={NUM_WORKERS}  '
          f'shards={NUM_SHARDS}  SR={SR}')
    print(f'  Tokens: {MIN_TOK}–{MAX_TOK} int16')
    print(f'  Fresh data per system · 1 epoch each')
    print('=' * W)

    results = {}
    all_times = {}
    all_sizes = {}
    for label, min_a, max_a in AUDIO_CONFIGS:
        rates, times, sz = bench_one_config(label, min_a, max_a)
        results[label] = rates
        all_times[label] = times
        all_sizes[label] = sz

    # ── Throughput summary ──
    print(f'\n{"=" * W}')
    print('  Throughput Summary  (speedup vs A = .wav + Torch)')
    print(f'{"=" * W}')
    print(f'  {"Duration":<12} '
          f'{"A .wav+T":>10} '
          f'{"B1 none+T":>11} '
          f'{"B2 flac+T":>11} '
          f'{"C1 none+C":>11} '
          f'{"C2 flac+C":>11} '
          f'{"B1/A":>7} {"B2/A":>7} {"C1/A":>7} {"C2/A":>7}')
    print(f'  {"─" * 12} '
          f'{"─" * 10} '
          f'{"─" * 11} '
          f'{"─" * 11} '
          f'{"─" * 11} '
          f'{"─" * 11} '
          f'{"─" * 7} {"─" * 7} {"─" * 7} {"─" * 7}')

    for label, _, _ in AUDIO_CONFIGS:
        ra, rb1, rb2, rc1, rc2 = results[label]
        s_b1 = rb1 / ra if ra > 0 else 0
        s_b2 = rb2 / ra if ra > 0 else 0
        s_c1 = rc1 / ra if ra > 0 else 0
        s_c2 = rc2 / ra if ra > 0 else 0
        print(f'  {label:<12} '
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
    print(f'  {"Duration":<12} '
          f'{"wav total":>12} '
          f'{"VVTK none":>12} '
          f'{"VVTK flac":>12} '
          f'{"flac/none":>10} '
          f'{"flac/wav":>10}')
    print(f'  {"─" * 12} '
          f'{"─" * 12} '
          f'{"─" * 12} '
          f'{"─" * 12} '
          f'{"─" * 10} '
          f'{"─" * 10}')

    for label, _, _ in AUDIO_CONFIGS:
        sz = all_sizes[label]
        sz_wav  = sz['wav']
        sz_none = sz['vvtk_none']
        sz_flac = sz['vvtk_flac']
        ratio_fn = sz_flac / sz_none if sz_none > 0 else 0
        ratio_fw = sz_flac / sz_wav  if sz_wav  > 0 else 0
        print(f'  {label:<12} '
              f'{_fmt_size(sz_wav):>12} '
              f'{_fmt_size(sz_none):>12} '
              f'{_fmt_size(sz_flac):>12} '
              f'{ratio_fn:>9.2f}x '
              f'{ratio_fw:>9.2f}x')

    print(f'{"=" * W}\n')

    cleanup()


if __name__ == '__main__':
    main()
