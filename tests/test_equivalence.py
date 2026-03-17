"""
Equivalence tests: verify that Systems B and C produce the same
numerical data as the baseline System A (.npy + PyTorch DataLoader).

Two scenarios matching the benchmark structure:
  1. Fixed-shape tensors (images + labels)       — benchmark_01 style
  2. Variable-length tensors (audio + tokens)     — benchmark_02 style

Each test:
  - Generates identical data for all 3 systems
  - Reads 1 epoch WITHOUT shuffle, with drop_last=True
  - Collects all samples, sorts by label/index to unify ordering
    (the C++ loader iterates in shard order, not global index order)
  - Asserts tensors match within a small tolerance (ABS_TOL)

Note: N must be a multiple of BATCH_SIZE so that drop_last=True yields the
same number of samples from every loader.  The C++ VVTKDataLoader always
drops the last batch when it is not full.
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
WORK_DIR   = os.path.join(os.path.dirname(__file__), 'temp_equiv')
ABS_TOL    = 1e-6         # tolerance for floating-point comparisons

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Test 1: Fixed-shape tensors (images + labels)
# ══════════════════════════════════════════════════════════════════════════════

IMG_SHAPE = (3, 32, 32)
LBL_SHAPE = (1,)


class NpyFixedDataset(Dataset):
    def __init__(self, folder, n):
        self.folder, self.n = folder, n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = torch.from_numpy(np.load(os.path.join(self.folder, f'{idx}_img.npy')))
        lbl = torch.from_numpy(np.load(os.path.join(self.folder, f'{idx}_lbl.npy')))
        return img, lbl


def _make_fixed_sample(rng, i):
    img = rng.integers(0, 256, size=IMG_SHAPE, dtype=np.uint8)
    lbl = np.array([i], dtype=np.int64)         # unique index as label
    return img, lbl


class TestFixedShapeEquivalence(unittest.TestCase):
    """Systems A, B, C must return identical fixed-shape tensors."""

    @classmethod
    def setUpClass(cls):
        cls.d = os.path.join(WORK_DIR, 'fixed')
        make_dir(cls.d)

        # ── Generate identical raw data ──
        rng = np.random.default_rng(42)
        cls.ref_imgs = []
        cls.ref_lbls = []
        for i in range(N):
            img, lbl = _make_fixed_sample(rng, i)
            cls.ref_imgs.append(img)
            cls.ref_lbls.append(lbl)

        # System A: .npy files
        folder_a = os.path.join(cls.d, 'a')
        os.makedirs(folder_a, exist_ok=True)
        for i in range(N):
            np.save(os.path.join(folder_a, f'{i}_img.npy'), cls.ref_imgs[i])
            np.save(os.path.join(folder_a, f'{i}_lbl.npy'), cls.ref_lbls[i])
        cls.folder_a = folder_a

        # System B: VVTK shards
        vvtk_b = os.path.join(cls.d, 'b', 'data')
        os.makedirs(os.path.dirname(vvtk_b), exist_ok=True)
        with VVTKDataset(vvtk_b, mode='wb', num_shards=4,
                         compression=['none', 'none']) as ds:
            for i in range(N):
                ds.add(i, cls.ref_imgs[i], cls.ref_lbls[i])
        cls.vvtk_b = vvtk_b

        # System C: separate VVTK shards
        vvtk_c = os.path.join(cls.d, 'c', 'data')
        os.makedirs(os.path.dirname(vvtk_c), exist_ok=True)
        with VVTKDataset(vvtk_c, mode='wb', num_shards=4,
                         compression=['none', 'none']) as ds:
            for i in range(N):
                ds.add(i, cls.ref_imgs[i], cls.ref_lbls[i])
        cls.vvtk_c = vvtk_c

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    # ── Collect data from each system, sorted by label ───────────────────────

    def _collect_A(self):
        ds = NpyFixedDataset(self.folder_a, N)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, drop_last=True)
        imgs, lbls = [], []
        for batch in loader:
            imgs.append(batch[0])
            lbls.append(batch[1])
        imgs, lbls = torch.cat(imgs), torch.cat(lbls)
        order = lbls.squeeze(-1).argsort()
        return imgs[order], lbls[order]

    def _collect_B(self):
        ds = VVTKDataset(self.vvtk_b, mode='rb', compression=['none', 'none'])
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, drop_last=True)
        imgs, lbls = [], []
        for batch in loader:
            imgs.append(batch[0])
            lbls.append(batch[1])
        ds.close()
        imgs, lbls = torch.cat(imgs), torch.cat(lbls)
        order = lbls.squeeze(-1).argsort()
        return imgs[order], lbls[order]

    def _collect_C(self):
        ds = VVTKDataset(self.vvtk_c, mode='rb', compression=['none', 'none'])
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

    # ── Tests ────────────────────────────────────────────────────────────────

    def test_B_vs_A_images(self):
        a_imgs, _ = self._collect_A()
        b_imgs, _ = self._collect_B()
        self.assertEqual(a_imgs.shape, b_imgs.shape)
        err = (a_imgs.float() - b_imgs.float()).abs().max().item()
        self.assertLessEqual(err, ABS_TOL,
                             f"System B images differ from baseline A (max err={err})")

    def test_B_vs_A_labels(self):
        _, a_lbls = self._collect_A()
        _, b_lbls = self._collect_B()
        self.assertEqual(a_lbls.shape, b_lbls.shape)
        err = (a_lbls.float() - b_lbls.float()).abs().max().item()
        self.assertLessEqual(err, ABS_TOL,
                             f"System B labels differ from baseline A (max err={err})")

    def test_C_vs_A_images(self):
        a_imgs, _ = self._collect_A()
        c_imgs, _ = self._collect_C()
        self.assertEqual(a_imgs.shape, c_imgs.shape)
        err = (a_imgs.float() - c_imgs.float()).abs().max().item()
        self.assertLessEqual(err, ABS_TOL,
                             f"System C images differ from baseline A (max err={err})")

    def test_C_vs_A_labels(self):
        _, a_lbls = self._collect_A()
        _, c_lbls = self._collect_C()
        self.assertEqual(a_lbls.shape, c_lbls.shape)
        err = (a_lbls.float() - c_lbls.float()).abs().max().item()
        self.assertLessEqual(err, ABS_TOL,
                             f"System C labels differ from baseline A (max err={err})")


# ══════════════════════════════════════════════════════════════════════════════
#  Test 2: Variable-length tensors (audio + tokens) with padding
# ══════════════════════════════════════════════════════════════════════════════

SR        = 16_000
MIN_AUDIO = 100
MAX_AUDIO = 800
MIN_TOK   = 5
MAX_TOK   = 50
PAD_AUDIO = MAX_AUDIO
PAD_TOK   = MAX_TOK


class NpyVarDataset(Dataset):
    """Baseline .npy dataset with padding.  Stores sample index for sorting."""

    def __init__(self, folder, n, pad_audio, pad_tok):
        self.folder, self.n = folder, n
        self.pad_audio, self.pad_tok = pad_audio, pad_tok

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        audio = np.load(os.path.join(self.folder, f'{idx}_audio.npy'))
        tokens = np.load(os.path.join(self.folder, f'{idx}_tok.npy'))
        index_arr = np.load(os.path.join(self.folder, f'{idx}_idx.npy'))

        alen = min(len(audio), self.pad_audio)
        tlen = min(len(tokens), self.pad_tok)

        pa = np.zeros(self.pad_audio, dtype=np.float32)
        pa[:alen] = audio[:alen]
        pt = np.zeros(self.pad_tok, dtype=np.int16)
        pt[:tlen] = tokens[:tlen]

        return (torch.from_numpy(pa), torch.tensor(alen, dtype=torch.int64),
                torch.from_numpy(pt), torch.tensor(tlen, dtype=torch.int64),
                torch.from_numpy(index_arr))  # carry index for sorting


def _make_var_sample(rng, i):
    rs = np.random.RandomState(seed=i)
    alen = rs.randint(MIN_AUDIO, MAX_AUDIO + 1)
    tlen = rs.randint(MIN_TOK, MAX_TOK + 1)
    freq = rng.uniform(200, 4000)
    t = np.arange(alen, dtype=np.float32) / SR
    waveform = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    tokens = rng.integers(0, 30000, size=tlen, dtype=np.int16)
    return waveform, tokens


class TestVariableLengthEquivalence(unittest.TestCase):
    """Systems A, B, C must return identical variable-length tensors (padded).

    For the variable-length case we write 3 tensors per sample:
      (audio, tokens, index)  where index = [i] int64
    so we can sort results from all systems into the same order.
    """

    @classmethod
    def setUpClass(cls):
        cls.d = os.path.join(WORK_DIR, 'var')
        make_dir(cls.d)

        rng = np.random.default_rng(42)
        cls.ref_audio, cls.ref_tokens = [], []
        for i in range(N):
            a, t = _make_var_sample(rng, i)
            cls.ref_audio.append(a)
            cls.ref_tokens.append(t)

        # System A: .npy files  (includes idx file for sorting)
        folder_a = os.path.join(cls.d, 'a')
        os.makedirs(folder_a, exist_ok=True)
        for i in range(N):
            np.save(os.path.join(folder_a, f'{i}_audio.npy'), cls.ref_audio[i])
            np.save(os.path.join(folder_a, f'{i}_tok.npy'), cls.ref_tokens[i])
            np.save(os.path.join(folder_a, f'{i}_idx.npy'),
                    np.array([i], dtype=np.int64))
        cls.folder_a = folder_a

        # System B: VVTK + Torch DL (with padding)
        # Stores (audio, tokens, index) — 3 tensors
        vvtk_b = os.path.join(cls.d, 'b', 'data')
        os.makedirs(os.path.dirname(vvtk_b), exist_ok=True)
        with VVTKDataset(vvtk_b, mode='wb', num_shards=4,
                         compression=['none', 'none', 'none']) as ds:
            for i in range(N):
                ds.add(i, cls.ref_audio[i], cls.ref_tokens[i],
                       np.array([i], dtype=np.int64))
        cls.vvtk_b = vvtk_b

        # System C: VVTK + C++ DL
        vvtk_c = os.path.join(cls.d, 'c', 'data')
        os.makedirs(os.path.dirname(vvtk_c), exist_ok=True)
        with VVTKDataset(vvtk_c, mode='wb', num_shards=4,
                         compression=['none', 'none', 'none']) as ds:
            for i in range(N):
                ds.add(i, cls.ref_audio[i], cls.ref_tokens[i],
                       np.array([i], dtype=np.int64))
        cls.vvtk_c = vvtk_c

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    # ── Collect (all return sorted by index) ─────────────────────────────────

    def _collect_A(self):
        ds = NpyVarDataset(self.folder_a, N, PAD_AUDIO, PAD_TOK)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, drop_last=True)
        audio, alen, tok, tlen, idx = [], [], [], [], []
        for batch in loader:
            audio.append(batch[0]); alen.append(batch[1])
            tok.append(batch[2]);   tlen.append(batch[3])
            idx.append(batch[4])
        audio = torch.cat(audio); alen = torch.cat(alen)
        tok   = torch.cat(tok);   tlen = torch.cat(tlen)
        idx   = torch.cat(idx).squeeze(-1)
        order = idx.argsort()
        return audio[order], alen[order], tok[order], tlen[order]

    def _collect_B(self):
        ds = VVTKDataset(self.vvtk_b, mode='rb',
                         compression=['none', 'none', 'none'],
                         fixed_shapes=[(PAD_AUDIO,), (PAD_TOK,), (1,)],
                         padding_values=[0.0, 0, 0])
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
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

    def _collect_C(self):
        ds = VVTKDataset(self.vvtk_c, mode='rb',
                         compression=['none', 'none', 'none'])
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

    # ── Tests ────────────────────────────────────────────────────────────────

    def test_B_vs_A_audio_data(self):
        a_audio, a_alen, _, _ = self._collect_A()
        b_audio, b_alen, _, _ = self._collect_B()
        self.assertEqual(a_audio.shape, b_audio.shape)
        for i in range(len(a_alen)):
            L = a_alen[i].item()
            err = (a_audio[i, :L].float() - b_audio[i, :L].float()).abs().max().item()
            self.assertLessEqual(err, ABS_TOL,
                                 f"Sample {i}: audio data mismatch (max err={err})")

    def test_B_vs_A_audio_lengths(self):
        _, a_alen, _, _ = self._collect_A()
        _, b_alen, _, _ = self._collect_B()
        err = (a_alen.float() - b_alen.float()).abs().max().item()
        self.assertLessEqual(err, ABS_TOL,
                             f"System B audio lengths differ from A (max err={err})")

    def test_B_vs_A_token_data(self):
        _, _, a_tok, a_tlen = self._collect_A()
        _, _, b_tok, b_tlen = self._collect_B()
        self.assertEqual(a_tok.shape, b_tok.shape)
        for i in range(len(a_tlen)):
            L = a_tlen[i].item()
            err = (a_tok[i, :L].long() - b_tok[i, :L].long()).abs().max().item()
            self.assertLessEqual(err, ABS_TOL,
                                 f"Sample {i}: token data mismatch (max err={err})")

    def test_B_vs_A_token_lengths(self):
        _, _, _, a_tlen = self._collect_A()
        _, _, _, b_tlen = self._collect_B()
        err = (a_tlen.float() - b_tlen.float()).abs().max().item()
        self.assertLessEqual(err, ABS_TOL,
                             f"System B token lengths differ from A (max err={err})")

    def test_C_vs_A_audio_data(self):
        a_audio, a_alen, _, _ = self._collect_A()
        c_audio, c_alen, _, _ = self._collect_C()
        self.assertEqual(a_audio.shape, c_audio.shape)
        for i in range(len(a_alen)):
            L = a_alen[i].item()
            err = (a_audio[i, :L].float() - c_audio[i, :L].float()).abs().max().item()
            self.assertLessEqual(err, ABS_TOL,
                                 f"Sample {i}: C vs A audio mismatch (max err={err})")

    def test_C_vs_A_audio_lengths(self):
        _, a_alen, _, _ = self._collect_A()
        _, c_alen, _, _ = self._collect_C()
        err = (a_alen.float() - c_alen.float()).abs().max().item()
        self.assertLessEqual(err, ABS_TOL,
                             f"System C audio lengths differ from A (max err={err})")

    def test_C_vs_A_token_data(self):
        _, _, a_tok, a_tlen = self._collect_A()
        _, _, c_tok, c_tlen = self._collect_C()
        self.assertEqual(a_tok.shape, c_tok.shape)
        for i in range(len(a_tlen)):
            L = a_tlen[i].item()
            err = (a_tok[i, :L].long() - c_tok[i, :L].long()).abs().max().item()
            self.assertLessEqual(err, ABS_TOL,
                                 f"Sample {i}: C vs A token mismatch (max err={err})")

    def test_C_vs_A_token_lengths(self):
        _, _, _, a_tlen = self._collect_A()
        _, _, _, c_tlen = self._collect_C()
        err = (a_tlen.float() - c_tlen.float()).abs().max().item()
        self.assertLessEqual(err, ABS_TOL,
                             f"System C token lengths differ from A (max err={err})")


# ══════════════════════════════════════════════════════════════════════════════
#  Cleanup
# ══════════════════════════════════════════════════════════════════════════════

def tearDownModule():
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
