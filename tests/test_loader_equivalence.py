"""
Loader equivalence tests: verify that the C++ VVTKDataLoader returns
exactly the same data as a standard torch DataLoader wrapping the same
VVTKDataset, for both uncompressed ('none') and zstd-compressed datasets.

Two scenarios × two compression modes = four test classes:

  1. Fixed-shape tensors (uint8 images + int64 labels)
     a) compression = ['none', 'none']
     b) compression = ['zstd', 'zstd']

  2. Variable-length tensors (float32 audio + int16 tokens)
     a) compression = ['none', 'none', 'none']
     b) compression = ['zstd', 'zstd', 'zstd']

For every class:
  - Write N samples to a VVTK dataset with the given compression.
  - Collect one full epoch from a torch DataLoader (reference).
  - Collect one full epoch from the C++ VVTKDataLoader (test).
  - Sort both by a unique label/index to normalise iteration order.
  - Assert exact match (tolerance ABS_TOL for float data, exact for int).

N is chosen as a multiple of BATCH_SIZE so that the C++ loader (which
always drops the last incomplete batch) returns the same sample count.
"""

import unittest
import os
import sys
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vvtk_dataset import VVTKDataset, VVTKDataLoader


# ── Config ───────────────────────────────────────────────────────────────────

N          = 192          # must be a multiple of BATCH_SIZE
BATCH_SIZE = 32
NUM_SHARDS = 4
WORK_DIR   = os.path.join(os.path.dirname(__file__), 'temp_loader_equiv')
ABS_TOL    = 1e-6

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Scenario 1: Fixed-shape tensors
# ══════════════════════════════════════════════════════════════════════════════

IMG_SHAPE = (3, 32, 32)
LBL_SHAPE = (1,)


class VVTKFixedTorchWrapper(Dataset):
    """Wraps VVTKDataset for a standard torch DataLoader (fixed shape)."""
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        img, lbl = self.ds[idx]
        return img, lbl


def _write_fixed_dataset(path, compression, ref_imgs, ref_lbls):
    """Write N fixed-shape samples to a VVTK dataset."""
    with VVTKDataset(path, mode='wb', num_shards=NUM_SHARDS,
                     compression=compression) as ds:
        for i in range(len(ref_imgs)):
            ds.add(i, ref_imgs[i], ref_lbls[i])


def _collect_fixed_torch(vvtk_path, compression):
    """Collect one epoch from torch DataLoader, sorted by label."""
    ds = VVTKDataset(vvtk_path, mode='rb', compression=compression)
    wrapper = VVTKFixedTorchWrapper(ds)
    loader = DataLoader(wrapper, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, drop_last=True)
    imgs, lbls = [], []
    for batch in loader:
        imgs.append(batch[0])
        lbls.append(batch[1])
    ds.close()
    imgs, lbls = torch.cat(imgs), torch.cat(lbls)
    order = lbls.squeeze(-1).argsort()
    return imgs[order], lbls[order]


def _collect_fixed_cpp(vvtk_path, compression):
    """Collect one epoch from C++ VVTKDataLoader, sorted by label."""
    ds = VVTKDataset(vvtk_path, mode='rb', compression=compression)
    loader = VVTKDataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=2,
        ring_size=4,
        shapes=[IMG_SHAPE, LBL_SHAPE],
        dtypes=[torch.uint8, torch.int64],
        shuffle=False,
    )
    imgs, lbls = [], []
    for batch in loader:
        (img_b, _), (lbl_b, _) = batch
        imgs.append(img_b)
        lbls.append(lbl_b)
    ds.close()
    imgs, lbls = torch.cat(imgs), torch.cat(lbls)
    order = lbls.squeeze(-1).argsort()
    return imgs[order], lbls[order]


# ── 1a: Fixed + none ────────────────────────────────────────────────────────

class TestFixedNone(unittest.TestCase):
    """C++ loader == torch loader for fixed-shape, compression='none'."""

    COMP = ['none', 'none']

    @classmethod
    def setUpClass(cls):
        cls.d = os.path.join(WORK_DIR, 'fixed_none')
        make_dir(cls.d)

        rng = np.random.default_rng(42)
        cls.ref_imgs = [rng.integers(0, 256, IMG_SHAPE, dtype=np.uint8)
                        for _ in range(N)]
        cls.ref_lbls = [np.array([i], dtype=np.int64) for i in range(N)]

        cls.vvtk_path = os.path.join(cls.d, 'data')
        _write_fixed_dataset(cls.vvtk_path, cls.COMP, cls.ref_imgs, cls.ref_lbls)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_images_match(self):
        t_imgs, _ = _collect_fixed_torch(self.vvtk_path, self.COMP)
        c_imgs, _ = _collect_fixed_cpp(self.vvtk_path, self.COMP)
        self.assertEqual(t_imgs.shape, c_imgs.shape)
        err = (t_imgs.float() - c_imgs.float()).abs().max().item()
        self.assertLessEqual(err, ABS_TOL,
                             f"Images differ (max err={err})")

    def test_labels_match(self):
        _, t_lbls = _collect_fixed_torch(self.vvtk_path, self.COMP)
        _, c_lbls = _collect_fixed_cpp(self.vvtk_path, self.COMP)
        self.assertEqual(t_lbls.shape, c_lbls.shape)
        self.assertTrue(torch.equal(t_lbls, c_lbls),
                        "Labels differ between torch and C++ loader")


# ── 1b: Fixed + zstd ────────────────────────────────────────────────────────

class TestFixedZstd(unittest.TestCase):
    """C++ loader == torch loader for fixed-shape, compression='zstd'."""

    COMP = ['zstd', 'zstd']

    @classmethod
    def setUpClass(cls):
        cls.d = os.path.join(WORK_DIR, 'fixed_zstd')
        make_dir(cls.d)

        rng = np.random.default_rng(42)
        cls.ref_imgs = [rng.integers(0, 256, IMG_SHAPE, dtype=np.uint8)
                        for _ in range(N)]
        cls.ref_lbls = [np.array([i], dtype=np.int64) for i in range(N)]

        cls.vvtk_path = os.path.join(cls.d, 'data')
        _write_fixed_dataset(cls.vvtk_path, cls.COMP, cls.ref_imgs, cls.ref_lbls)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_images_match(self):
        t_imgs, _ = _collect_fixed_torch(self.vvtk_path, self.COMP)
        c_imgs, _ = _collect_fixed_cpp(self.vvtk_path, self.COMP)
        self.assertEqual(t_imgs.shape, c_imgs.shape)
        err = (t_imgs.float() - c_imgs.float()).abs().max().item()
        self.assertLessEqual(err, ABS_TOL,
                             f"Images differ (max err={err})")

    def test_labels_match(self):
        _, t_lbls = _collect_fixed_torch(self.vvtk_path, self.COMP)
        _, c_lbls = _collect_fixed_cpp(self.vvtk_path, self.COMP)
        self.assertEqual(t_lbls.shape, c_lbls.shape)
        self.assertTrue(torch.equal(t_lbls, c_lbls),
                        "Labels differ between torch and C++ loader")


# ── 1c: Fixed + mixed (image=zstd, label=none) ──────────────────────────────

class TestFixedMixed(unittest.TestCase):
    """C++ loader == torch loader for fixed-shape, mixed compression."""

    COMP = ['zstd', 'none']

    @classmethod
    def setUpClass(cls):
        cls.d = os.path.join(WORK_DIR, 'fixed_mixed')
        make_dir(cls.d)

        rng = np.random.default_rng(42)
        cls.ref_imgs = [rng.integers(0, 256, IMG_SHAPE, dtype=np.uint8)
                        for _ in range(N)]
        cls.ref_lbls = [np.array([i], dtype=np.int64) for i in range(N)]

        cls.vvtk_path = os.path.join(cls.d, 'data')
        _write_fixed_dataset(cls.vvtk_path, cls.COMP, cls.ref_imgs, cls.ref_lbls)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_images_match(self):
        t_imgs, _ = _collect_fixed_torch(self.vvtk_path, self.COMP)
        c_imgs, _ = _collect_fixed_cpp(self.vvtk_path, self.COMP)
        self.assertEqual(t_imgs.shape, c_imgs.shape)
        err = (t_imgs.float() - c_imgs.float()).abs().max().item()
        self.assertLessEqual(err, ABS_TOL,
                             f"Images differ (max err={err})")

    def test_labels_match(self):
        _, t_lbls = _collect_fixed_torch(self.vvtk_path, self.COMP)
        _, c_lbls = _collect_fixed_cpp(self.vvtk_path, self.COMP)
        self.assertEqual(t_lbls.shape, c_lbls.shape)
        self.assertTrue(torch.equal(t_lbls, c_lbls),
                        "Labels differ between torch and C++ loader")


# ══════════════════════════════════════════════════════════════════════════════
#  Scenario 2: Variable-length tensors (audio + tokens + index)
# ══════════════════════════════════════════════════════════════════════════════

SR        = 16_000
MIN_AUDIO = 100
MAX_AUDIO = 800
MIN_TOK   = 5
MAX_TOK   = 50
PAD_AUDIO = MAX_AUDIO
PAD_TOK   = MAX_TOK


def _make_var_sample(rng, i):
    rs = np.random.RandomState(seed=i)
    alen = rs.randint(MIN_AUDIO, MAX_AUDIO + 1)
    tlen = rs.randint(MIN_TOK, MAX_TOK + 1)
    freq = rng.uniform(200, 4000)
    t = np.arange(alen, dtype=np.float32) / SR
    waveform = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    tokens = rng.integers(0, 30000, size=tlen, dtype=np.int16)
    return waveform, tokens


def _write_var_dataset(path, compression, ref_audio, ref_tokens):
    """Write N variable-length samples (audio, tokens, index) to VVTK."""
    with VVTKDataset(path, mode='wb', num_shards=NUM_SHARDS,
                     compression=compression) as ds:
        for i in range(len(ref_audio)):
            ds.add(i, ref_audio[i], ref_tokens[i],
                   np.array([i], dtype=np.int64))


class VVTKVarTorchWrapper(Dataset):
    """Wraps a padded VVTKDataset for a standard torch DataLoader."""
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        return self.ds[idx]


def _collect_var_torch(vvtk_path, compression):
    """Collect one epoch from torch DataLoader (variable-length, padded), sorted by index."""
    ds = VVTKDataset(vvtk_path, mode='rb', compression=compression,
                     fixed_shapes=[(PAD_AUDIO,), (PAD_TOK,), (1,)],
                     padding_values=[0.0, 0, 0])
    wrapper = VVTKVarTorchWrapper(ds)
    loader = DataLoader(wrapper, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, drop_last=True)
    audio, alen, tok, tlen, idx = [], [], [], [], []
    for batch in loader:
        (a, al), (t, tl), (ix, _) = batch
        audio.append(a); alen.append(al)
        tok.append(t);   tlen.append(tl)
        idx.append(ix)
    ds.close()
    audio = torch.cat(audio); alen = torch.cat(alen)
    tok   = torch.cat(tok);   tlen = torch.cat(tlen)
    idx   = torch.cat(idx).squeeze(-1)
    order = idx.argsort()
    return audio[order], alen[order], tok[order], tlen[order]


def _collect_var_cpp(vvtk_path, compression):
    """Collect one epoch from C++ VVTKDataLoader (variable-length), sorted by index."""
    ds = VVTKDataset(vvtk_path, mode='rb', compression=compression)
    loader = VVTKDataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=2,
        ring_size=4,
        shapes=[(PAD_AUDIO,), (PAD_TOK,), (1,)],
        dtypes=[torch.float32, torch.int16, torch.int64],
        padding_values=[0.0, 0, 0],
        shuffle=False,
    )
    audio, alen, tok, tlen, idx = [], [], [], [], []
    for batch in loader:
        (a, al), (t, tl), (ix, _) = batch
        audio.append(a); alen.append(al)
        tok.append(t);   tlen.append(tl)
        idx.append(ix)
    ds.close()
    audio = torch.cat(audio); alen = torch.cat(alen)
    tok   = torch.cat(tok);   tlen = torch.cat(tlen)
    idx   = torch.cat(idx).squeeze(-1)
    order = idx.argsort()
    return audio[order], alen[order], tok[order], tlen[order]


# ── 2a: Variable + none ─────────────────────────────────────────────────────

class TestVariableNone(unittest.TestCase):
    """C++ loader == torch loader for variable-length, compression='none'."""

    COMP = ['none', 'none', 'none']

    @classmethod
    def setUpClass(cls):
        cls.d = os.path.join(WORK_DIR, 'var_none')
        make_dir(cls.d)

        rng = np.random.default_rng(42)
        cls.ref_audio, cls.ref_tokens = [], []
        for i in range(N):
            a, t = _make_var_sample(rng, i)
            cls.ref_audio.append(a)
            cls.ref_tokens.append(t)

        cls.vvtk_path = os.path.join(cls.d, 'data')
        _write_var_dataset(cls.vvtk_path, cls.COMP, cls.ref_audio, cls.ref_tokens)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_audio_data_match(self):
        t_audio, t_alen, _, _ = _collect_var_torch(self.vvtk_path, self.COMP)
        c_audio, c_alen, _, _ = _collect_var_cpp(self.vvtk_path, self.COMP)
        self.assertEqual(t_audio.shape, c_audio.shape)
        for i in range(len(t_alen)):
            L = t_alen[i].item()
            err = (t_audio[i, :L] - c_audio[i, :L]).abs().max().item()
            self.assertLessEqual(err, ABS_TOL,
                                 f"Sample {i}: audio mismatch (max err={err})")

    def test_audio_lengths_match(self):
        _, t_alen, _, _ = _collect_var_torch(self.vvtk_path, self.COMP)
        _, c_alen, _, _ = _collect_var_cpp(self.vvtk_path, self.COMP)
        self.assertTrue(torch.equal(t_alen, c_alen),
                        "Audio lengths differ between torch and C++ loader")

    def test_token_data_match(self):
        _, _, t_tok, t_tlen = _collect_var_torch(self.vvtk_path, self.COMP)
        _, _, c_tok, c_tlen = _collect_var_cpp(self.vvtk_path, self.COMP)
        self.assertEqual(t_tok.shape, c_tok.shape)
        for i in range(len(t_tlen)):
            L = t_tlen[i].item()
            err = (t_tok[i, :L].long() - c_tok[i, :L].long()).abs().max().item()
            self.assertLessEqual(err, ABS_TOL,
                                 f"Sample {i}: token mismatch (max err={err})")

    def test_token_lengths_match(self):
        _, _, _, t_tlen = _collect_var_torch(self.vvtk_path, self.COMP)
        _, _, _, c_tlen = _collect_var_cpp(self.vvtk_path, self.COMP)
        self.assertTrue(torch.equal(t_tlen, c_tlen),
                        "Token lengths differ between torch and C++ loader")


# ── 2b: Variable + zstd ─────────────────────────────────────────────────────

class TestVariableZstd(unittest.TestCase):
    """C++ loader == torch loader for variable-length, compression='zstd'."""

    COMP = ['zstd', 'zstd', 'zstd']

    @classmethod
    def setUpClass(cls):
        cls.d = os.path.join(WORK_DIR, 'var_zstd')
        make_dir(cls.d)

        rng = np.random.default_rng(42)
        cls.ref_audio, cls.ref_tokens = [], []
        for i in range(N):
            a, t = _make_var_sample(rng, i)
            cls.ref_audio.append(a)
            cls.ref_tokens.append(t)

        cls.vvtk_path = os.path.join(cls.d, 'data')
        _write_var_dataset(cls.vvtk_path, cls.COMP, cls.ref_audio, cls.ref_tokens)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_audio_data_match(self):
        t_audio, t_alen, _, _ = _collect_var_torch(self.vvtk_path, self.COMP)
        c_audio, c_alen, _, _ = _collect_var_cpp(self.vvtk_path, self.COMP)
        self.assertEqual(t_audio.shape, c_audio.shape)
        for i in range(len(t_alen)):
            L = t_alen[i].item()
            err = (t_audio[i, :L] - c_audio[i, :L]).abs().max().item()
            self.assertLessEqual(err, ABS_TOL,
                                 f"Sample {i}: audio mismatch (max err={err})")

    def test_audio_lengths_match(self):
        _, t_alen, _, _ = _collect_var_torch(self.vvtk_path, self.COMP)
        _, c_alen, _, _ = _collect_var_cpp(self.vvtk_path, self.COMP)
        self.assertTrue(torch.equal(t_alen, c_alen),
                        "Audio lengths differ between torch and C++ loader")

    def test_token_data_match(self):
        _, _, t_tok, t_tlen = _collect_var_torch(self.vvtk_path, self.COMP)
        _, _, c_tok, c_tlen = _collect_var_cpp(self.vvtk_path, self.COMP)
        self.assertEqual(t_tok.shape, c_tok.shape)
        for i in range(len(t_tlen)):
            L = t_tlen[i].item()
            err = (t_tok[i, :L].long() - c_tok[i, :L].long()).abs().max().item()
            self.assertLessEqual(err, ABS_TOL,
                                 f"Sample {i}: token mismatch (max err={err})")

    def test_token_lengths_match(self):
        _, _, _, t_tlen = _collect_var_torch(self.vvtk_path, self.COMP)
        _, _, _, c_tlen = _collect_var_cpp(self.vvtk_path, self.COMP)
        self.assertTrue(torch.equal(t_tlen, c_tlen),
                        "Token lengths differ between torch and C++ loader")


# ── 2c: Variable + mixed (audio=zstd, tokens=none, index=none) ──────────────

class TestVariableMixed(unittest.TestCase):
    """C++ loader == torch loader for variable-length, mixed compression."""

    COMP = ['zstd', 'none', 'none']

    @classmethod
    def setUpClass(cls):
        cls.d = os.path.join(WORK_DIR, 'var_mixed')
        make_dir(cls.d)

        rng = np.random.default_rng(42)
        cls.ref_audio, cls.ref_tokens = [], []
        for i in range(N):
            a, t = _make_var_sample(rng, i)
            cls.ref_audio.append(a)
            cls.ref_tokens.append(t)

        cls.vvtk_path = os.path.join(cls.d, 'data')
        _write_var_dataset(cls.vvtk_path, cls.COMP, cls.ref_audio, cls.ref_tokens)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_audio_data_match(self):
        t_audio, t_alen, _, _ = _collect_var_torch(self.vvtk_path, self.COMP)
        c_audio, c_alen, _, _ = _collect_var_cpp(self.vvtk_path, self.COMP)
        self.assertEqual(t_audio.shape, c_audio.shape)
        for i in range(len(t_alen)):
            L = t_alen[i].item()
            err = (t_audio[i, :L] - c_audio[i, :L]).abs().max().item()
            self.assertLessEqual(err, ABS_TOL,
                                 f"Sample {i}: audio mismatch (max err={err})")

    def test_audio_lengths_match(self):
        _, t_alen, _, _ = _collect_var_torch(self.vvtk_path, self.COMP)
        _, c_alen, _, _ = _collect_var_cpp(self.vvtk_path, self.COMP)
        self.assertTrue(torch.equal(t_alen, c_alen),
                        "Audio lengths differ between torch and C++ loader")

    def test_token_data_match(self):
        _, _, t_tok, t_tlen = _collect_var_torch(self.vvtk_path, self.COMP)
        _, _, c_tok, c_tlen = _collect_var_cpp(self.vvtk_path, self.COMP)
        self.assertEqual(t_tok.shape, c_tok.shape)
        for i in range(len(t_tlen)):
            L = t_tlen[i].item()
            err = (t_tok[i, :L].long() - c_tok[i, :L].long()).abs().max().item()
            self.assertLessEqual(err, ABS_TOL,
                                 f"Sample {i}: token mismatch (max err={err})")

    def test_token_lengths_match(self):
        _, _, _, t_tlen = _collect_var_torch(self.vvtk_path, self.COMP)
        _, _, _, c_tlen = _collect_var_cpp(self.vvtk_path, self.COMP)
        self.assertTrue(torch.equal(t_tlen, c_tlen),
                        "Token lengths differ between torch and C++ loader")


# ══════════════════════════════════════════════════════════════════════════════
#  Cleanup
# ══════════════════════════════════════════════════════════════════════════════

def tearDownModule():
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
