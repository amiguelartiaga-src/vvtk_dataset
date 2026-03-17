"""
Comprehensive tests for VVTKDataset and VVTKDataLoader.

Covers:
  - 1 to 4 tensors per sample
  - 1D, 2D, 3D shapes
  - All dtypes: uint8, int8, int16, int32, int64, float16, float32, float64
  - Fixed-length tensors
  - Variable-length tensors with padding
  - Different padding values
  - Dataset-only raw access
  - Dataset + PyTorch DataLoader
  - Dataset + C++ VVTKDataLoader
  - Different shard counts (1, 4, 8)
  - Edge cases: single sample, batch > dataset

Output: prints a dot per passed sub-check for compact progress.
"""

import unittest
import os
import sys
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from vvtk_dataset import VVTKDataset, VVTKDataLoader


# ── Helpers ──────────────────────────────────────────────────────────────────

def dot():
    sys.stdout.write('.')
    sys.stdout.flush()

def make_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)

def write_dataset(path, num_shards, compression, samples_fn, N):
    """Write a dataset and ensure files are flushed (NFS-safe)."""
    with VVTKDataset(path, mode='wb', num_shards=num_shards, compression=compression) as ds:
        for i in range(N):
            tensors = samples_fn(i)
            ds.add(i, *tensors)
    # Force directory metadata sync for NFS mounts
    d = os.path.dirname(path) or '.'
    try:
        fd = os.open(d, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass


# ── 1. Dtypes ────────────────────────────────────────────────────────────────

class TestAllDtypes(unittest.TestCase):
    """Write and read a single tensor for every supported dtype."""

    DTYPES = [
        (np.uint8,   torch.uint8,   0),
        (np.int8,    torch.int8,    1),
        (np.int16,   torch.int16,   2),
        (np.int32,   torch.int32,   3),
        (np.int64,   torch.int64,   4),
        (np.float16, torch.float16, 10),
        (np.float32, torch.float32, 11),
        (np.float64, torch.float64, 12),
    ]

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_dtypes'
        make_dir(cls.work_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_each_dtype(self):
        """Write/read 10 samples for every dtype (1 tensor per sample)."""
        N = 10
        shape = (64,)
        for np_dt, torch_dt, _ in self.DTYPES:
            subdir = os.path.join(self.work_dir, f'dtype_{np_dt.__name__}')
            os.makedirs(subdir, exist_ok=True)
            path = os.path.join(subdir, 'data')
            write_dataset(path, 1, ['none'],
                          lambda i, dt=np_dt: (np.full(shape, i % 127, dtype=dt),), N)
            rd = VVTKDataset(path, mode='rb', compression=['none'])
            for i in range(N):
                (t,) = rd[i]
                self.assertEqual(t.dtype, torch_dt)
                self.assertEqual(t.shape, shape)
                self.assertTrue(torch.all(t == (i % 127)))
                dot()
            rd.close()


# ── 2. Multiple tensors per sample ───────────────────────────────────────────

class TestMultipleTensors(unittest.TestCase):
    """1, 2, 3, and 4 tensors per sample — fixed shapes."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_multi'
        make_dir(cls.work_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def _roundtrip(self, n_tensors, N=50):
        path = os.path.join(self.work_dir, f'multi_{n_tensors}')
        comp = ['none'] * n_tensors
        shapes = [(16 * (k + 1),) for k in range(n_tensors)]
        np_dts = [np.float32] * n_tensors

        with VVTKDataset(path, mode='wb', num_shards=2, compression=comp) as ds:
            for i in range(N):
                tensors = [np.full(shapes[k], float(i + k), dtype=np_dts[k]) for k in range(n_tensors)]
                ds.add(i, *tensors)

        ds = VVTKDataset(path, mode='rb', compression=comp)
        for i in range(N):
            result = ds[i]
            self.assertEqual(len(result), n_tensors)
            for k in range(n_tensors):
                self.assertEqual(result[k].shape, shapes[k])
                self.assertTrue(torch.all(result[k] == float(i + k)))
                dot()
        ds.close()

    def test_1_tensor(self):
        self._roundtrip(1)

    def test_2_tensors(self):
        self._roundtrip(2)

    def test_3_tensors(self):
        self._roundtrip(3)

    def test_4_tensors(self):
        self._roundtrip(4)


# ── 3. Various shapes (1D, 2D, 3D) ──────────────────────────────────────────

class TestVariousShapes(unittest.TestCase):
    """Test tensors with 1D, 2D, and 3D shapes."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_shapes'
        make_dir(cls.work_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def _test_shape(self, shape, name, N=30):
        path = os.path.join(self.work_dir, name)
        with VVTKDataset(path, mode='wb', num_shards=2, compression=['none']) as ds:
            for i in range(N):
                t = np.full(shape, float(i), dtype=np.float32)
                ds.add(i, t)
        ds = VVTKDataset(path, mode='rb', compression=['none'])
        for i in range(N):
            (t,) = ds[i]
            self.assertEqual(tuple(t.shape), shape)
            self.assertTrue(torch.all(t == float(i)))
            dot()
        ds.close()

    def test_1d_small(self):
        self._test_shape((8,), '1d_small')

    def test_1d_large(self):
        self._test_shape((4096,), '1d_large')

    def test_2d(self):
        self._test_shape((32, 64), '2d')

    def test_3d(self):
        self._test_shape((3, 16, 16), '3d')

    def test_3d_image_like(self):
        self._test_shape((3, 112, 112), '3d_img')


# ── 4. Mixed dtypes in a multi-tensor sample ────────────────────────────────

class TestMixedDtypes(unittest.TestCase):
    """Sample = (float32 image, int64 label, int16 tokens)."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_mixed'
        make_dir(cls.work_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_mixed_dtypes_roundtrip(self):
        N = 40
        path = os.path.join(self.work_dir, 'mixed')
        comp = ['none', 'none', 'none']

        with VVTKDataset(path, mode='wb', num_shards=2, compression=comp) as ds:
            for i in range(N):
                img = np.full((3, 8, 8), float(i), dtype=np.float32)
                lbl = np.array([i], dtype=np.int64)
                tok = np.full((20,), i % 1000, dtype=np.int16)
                ds.add(i, img, lbl, tok)

        ds = VVTKDataset(path, mode='rb', compression=comp)
        for i in range(N):
            img, lbl, tok = ds[i]
            self.assertEqual(img.dtype, torch.float32)
            self.assertEqual(lbl.dtype, torch.int64)
            self.assertEqual(tok.dtype, torch.int16)
            self.assertTrue(torch.all(img == float(i)))
            self.assertEqual(lbl.item(), i)
            self.assertTrue(torch.all(tok == (i % 1000)))
            dot()
        ds.close()


# ── 5. Sharding ──────────────────────────────────────────────────────────────

class TestSharding(unittest.TestCase):
    """Write with different shard counts and verify reads."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_shards'
        make_dir(cls.work_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def _test_shard_count(self, num_shards, N=100):
        path = os.path.join(self.work_dir, f'shards_{num_shards}')
        with VVTKDataset(path, mode='wb', num_shards=num_shards, compression=['none', 'none']) as ds:
            for i in range(N):
                data = np.full((16,), float(i), dtype=np.float32)
                label = np.array([i], dtype=np.int64)
                ds.add(i, data, label)
        ds = VVTKDataset(path, mode='rb', compression=['none', 'none'])
        self.assertEqual(len(ds), N)
        for i in range(N):
            data, label = ds[i]
            self.assertTrue(torch.all(data == float(i)))
            self.assertEqual(label.item(), i)
            dot()
        ds.close()

    def test_1_shard(self):
        self._test_shard_count(1)

    def test_4_shards(self):
        self._test_shard_count(4)

    def test_8_shards(self):
        self._test_shard_count(8)


# ── 6. Variable-length + padding (dataset side) ─────────────────────────────

class TestVariableLengthPadding(unittest.TestCase):
    """Variable-length 1D tensors with dataset-side padding."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_varpad'
        make_dir(cls.work_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_padding_with_default_value(self):
        """Pad with 0.0 (default)."""
        N, max_len = 60, 200
        path = os.path.join(self.work_dir, 'pad_default')
        with VVTKDataset(path, mode='wb', num_shards=2, compression=['none']) as ds:
            for i in range(N):
                length = 50 + (i * 3) % 150  # 50..199
                t = np.full((length,), float(i), dtype=np.float32)
                ds.add(i, t)

        ds = VVTKDataset(path, mode='rb', compression=['none'],
                         fixed_shapes=[(max_len,)], padding_values=[0.0])
        for i in range(N):
            (data, real_len), = ds[i]
            length = 50 + (i * 3) % 150
            self.assertEqual(real_len.item(), length)
            self.assertEqual(data.shape[0], max_len)
            self.assertTrue(torch.all(data[:length] == float(i)))
            if length < max_len:
                self.assertTrue(torch.all(data[length:] == 0.0))
            dot()
        ds.close()

    def test_padding_with_custom_value(self):
        """Pad with -1."""
        N, max_len = 40, 100
        subdir = os.path.join(self.work_dir, 'pad_custom')
        os.makedirs(subdir, exist_ok=True)
        path = os.path.join(subdir, 'data')
        write_dataset(path, 1, ['none'],
                      lambda i: (np.full((10 + i % 90,), float(i), dtype=np.float32),), N)

        ds = VVTKDataset(path, mode='rb', compression=['none'],
                         fixed_shapes=[(max_len,)], padding_values=[-1.0])
        for i in range(N):
            (data, real_len), = ds[i]
            length = 10 + i % 90
            self.assertEqual(real_len.item(), length)
            self.assertTrue(torch.all(data[:length] == float(i)))
            if length < max_len:
                self.assertTrue(torch.all(data[length:] == -1.0))
            dot()
        ds.close()

    def test_truncation(self):
        """Data longer than max_len gets truncated."""
        N, max_len = 20, 50
        subdir = os.path.join(self.work_dir, 'trunc')
        os.makedirs(subdir, exist_ok=True)
        path = os.path.join(subdir, 'data')
        write_dataset(path, 1, ['none'],
                      lambda i: (np.full((80,), float(i), dtype=np.float32),), N)

        ds = VVTKDataset(path, mode='rb', compression=['none'],
                         fixed_shapes=[(max_len,)], padding_values=[0.0])
        for i in range(N):
            (data, real_len), = ds[i]
            self.assertEqual(real_len.item(), max_len)
            self.assertEqual(data.shape[0], max_len)
            self.assertTrue(torch.all(data == float(i)))
            dot()
        ds.close()

    def test_variable_multi_tensor_padding(self):
        """Two variable-length tensors with different padding values."""
        N, max1, max2 = 50, 128, 64
        path = os.path.join(self.work_dir, 'pad_multi')
        with VVTKDataset(path, mode='wb', num_shards=2,
                         compression=['none', 'none']) as ds:
            for i in range(N):
                l1 = 20 + i % 100
                l2 = 5 + i % 55
                t1 = np.full((l1,), float(i), dtype=np.float32)
                t2 = np.full((l2,), i, dtype=np.int32)
                ds.add(i, t1, t2)

        ds = VVTKDataset(path, mode='rb', compression=['none', 'none'],
                         fixed_shapes=[(max1,), (max2,)],
                         padding_values=[0.0, -1])
        for i in range(N):
            (d1, len1), (d2, len2) = ds[i]
            l1 = 20 + i % 100
            l2 = 5 + i % 55
            self.assertEqual(len1.item(), l1)
            self.assertEqual(len2.item(), l2)
            self.assertEqual(d1.shape[0], max1)
            self.assertEqual(d2.shape[0], max2)
            self.assertTrue(torch.all(d1[:l1] == float(i)))
            self.assertTrue(torch.all(d2[:l2] == i))
            if l1 < max1:
                self.assertTrue(torch.all(d1[l1:] == 0.0))
            if l2 < max2:
                self.assertTrue(torch.all(d2[l2:] == -1))
            dot()
        ds.close()


# ── 7. PyTorch DataLoader (fixed) ───────────────────────────────────────────

class TestPyTorchDataLoaderFixed(unittest.TestCase):
    """Dataset + standard torch DataLoader with fixed-shape tensors."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_torch_fixed'
        cls.N = 100
        cls.batch_size = 16
        cls.path = os.path.join(cls.work_dir, 'ds')
        make_dir(cls.work_dir)

        with VVTKDataset(cls.path, mode='wb', num_shards=4,
                         compression=['none', 'none']) as ds:
            for i in range(cls.N):
                img = np.full((3, 8, 8), float(i), dtype=np.float32)
                lbl = np.array([i], dtype=np.int64)
                ds.add(i, img, lbl)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_sequential_batches(self):
        ds = VVTKDataset(self.path, mode='rb', compression=['none', 'none'])
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        seen = 0
        for batch in loader:
            imgs, lbls = batch
            bs = imgs.shape[0]
            self.assertEqual(imgs.shape, (bs, 3, 8, 8))
            self.assertEqual(lbls.shape, (bs, 1))
            seen += bs
            dot()
        self.assertEqual(seen, self.N)
        ds.close()

    def test_shuffled_batches(self):
        ds = VVTKDataset(self.path, mode='rb', compression=['none', 'none'])
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        all_labels = []
        for batch in loader:
            _, lbls = batch
            all_labels.append(lbls.flatten())
            dot()
        all_labels = torch.cat(all_labels).sort().values
        self.assertTrue(torch.equal(all_labels, torch.arange(self.N)))
        ds.close()

    def test_multiworker(self):
        ds = VVTKDataset(self.path, mode='rb', compression=['none', 'none'])
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=2)
        seen = 0
        for batch in loader:
            imgs, lbls = batch
            seen += imgs.shape[0]
            dot()
        self.assertEqual(seen, self.N)
        ds.close()


# ── 8. PyTorch DataLoader (variable-length, padded) ─────────────────────────

class TestPyTorchDataLoaderVariable(unittest.TestCase):
    """Dataset + torch DataLoader with variable-length padded tensors."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_torch_var'
        cls.N = 80
        cls.batch_size = 16
        cls.max_len = 256
        cls.path = os.path.join(cls.work_dir, 'ds')
        make_dir(cls.work_dir)

        with VVTKDataset(cls.path, mode='wb', num_shards=2,
                         compression=['none', 'none']) as ds:
            for i in range(cls.N):
                length = 30 + (i * 7) % 220
                audio = np.full((length,), float(i), dtype=np.float32)
                label = np.array([i], dtype=np.int64)
                ds.add(i, audio, label)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_padded_batches(self):
        ds = VVTKDataset(self.path, mode='rb', compression=['none', 'none'],
                         fixed_shapes=[(self.max_len,), (1,)],
                         padding_values=[0.0, 0])
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        seen = 0
        for batch in loader:
            (audio, audio_len), (label, label_len) = batch
            bs = audio.shape[0]
            self.assertEqual(audio.shape, (bs, self.max_len))
            self.assertEqual(label.shape, (bs, 1))
            for b in range(bs):
                idx = seen + b
                exp_len = 30 + (idx * 7) % 220
                self.assertEqual(audio_len[b].item(), exp_len)
                self.assertTrue(torch.all(audio[b, :exp_len] == float(idx)))
                if exp_len < self.max_len:
                    self.assertTrue(torch.all(audio[b, exp_len:] == 0.0))
            seen += bs
            dot()
        self.assertEqual(seen, self.N)
        ds.close()


# ── 9. C++ VVTKDataLoader (fixed) ───────────────────────────────────────────

class TestCppDataLoaderFixed(unittest.TestCase):
    """Dataset + C++ VVTKDataLoader with fixed-shape tensors."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_cpp_fixed'
        cls.N = 100
        cls.batch_size = 16
        cls.data_shape = (3, 8, 8)
        cls.label_shape = (1,)
        cls.path = os.path.join(cls.work_dir, 'ds')
        make_dir(cls.work_dir)

        with VVTKDataset(cls.path, mode='wb', num_shards=4,
                         compression=['none', 'none']) as ds:
            for i in range(cls.N):
                img = np.full(cls.data_shape, float(i), dtype=np.float32)
                lbl = np.array([i], dtype=np.int64)
                ds.add(i, img, lbl)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_iterate_all(self):
        ds = VVTKDataset(self.path, mode='rb', compression=['none', 'none'])
        loader = VVTKDataLoader(ds, batch_size=self.batch_size, num_workers=2,
                                ring_size=4,
                                shapes=[self.data_shape, self.label_shape],
                                dtypes=[torch.float32, torch.int64])
        seen = 0
        for batch in loader:
            (data, data_lens), (label, label_lens) = batch
            bs = data.shape[0]
            self.assertEqual(data.shape[1:], self.data_shape)
            self.assertEqual(label.shape[1:], self.label_shape)
            data_vals = data[:, 0, 0, 0].long()
            label_vals = label.flatten()
            self.assertTrue(torch.equal(data_vals, label_vals))
            seen += bs
            dot()
        # C++ loader always drops last batch if not full
        expected = (self.N // self.batch_size) * self.batch_size
        self.assertEqual(seen, expected)
        ds.close()

    def test_integrity_values(self):
        ds = VVTKDataset(self.path, mode='rb', compression=['none', 'none'])
        loader = VVTKDataLoader(ds, batch_size=self.batch_size, num_workers=2,
                                ring_size=4,
                                shapes=[self.data_shape, self.label_shape],
                                dtypes=[torch.float32, torch.int64])
        all_labels = []
        for batch in loader:
            (data, data_lens), (label, label_lens) = batch
            bs = data.shape[0]
            for b in range(bs):
                if data_lens[b].item() > 0:
                    all_labels.append(label[b].flatten()[0].item())
            dot()
        all_labels = sorted(set(all_labels))
        # C++ loader always drops last batch, so some samples may be missing
        expected_count = (self.N // self.batch_size) * self.batch_size
        self.assertEqual(len(all_labels), expected_count)
        ds.close()


# ── 10. C++ VVTKDataLoader (variable-length) ────────────────────────────────

class TestCppDataLoaderVariable(unittest.TestCase):
    """Dataset + C++ VVTKDataLoader with variable-length padded tensors."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_cpp_var'
        cls.N = 80
        cls.batch_size = 16
        cls.max_shape = (300,)
        cls.label_shape = (1,)
        cls.path = os.path.join(cls.work_dir, 'ds')
        make_dir(cls.work_dir)

        with VVTKDataset(cls.path, mode='wb', num_shards=2,
                         compression=['none', 'none']) as ds:
            for i in range(cls.N):
                length = 20 + (i * 11) % 280
                audio = np.full((length,), float(i), dtype=np.float32)
                label = np.array([i], dtype=np.int64)
                ds.add(i, audio, label)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_iterate_with_padding(self):
        ds = VVTKDataset(self.path, mode='rb', compression=['none', 'none'])
        loader = VVTKDataLoader(ds, batch_size=self.batch_size, num_workers=2,
                                ring_size=4,
                                shapes=[self.max_shape, self.label_shape],
                                dtypes=[torch.float32, torch.int64],
                                padding_values=[0.0, 0.0])
        real_count = 0
        for batch in loader:
            (audio, audio_lens), (label, label_lens) = batch
            bs = audio.shape[0]
            self.assertEqual(audio.shape[1:], self.max_shape)
            for b in range(bs):
                rl = audio_lens[b].item()
                if rl == 0:
                    continue
                idx = label[b].flatten()[0].item()
                exp_len = 20 + (idx * 11) % 280
                self.assertEqual(rl, exp_len)
                self.assertTrue(torch.all(audio[b, :rl] == float(idx)))
                real_count += 1
            dot()
        self.assertEqual(real_count, self.N)
        ds.close()


# ── 11. C++ VVTKDataLoader with shuffle ──────────────────────────────────────

class TestCppDataLoaderShuffle(unittest.TestCase):
    """Verify shuffle mode returns all samples in a different order."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_cpp_shuffle'
        cls.N = 64
        cls.batch_size = 16
        cls.path = os.path.join(cls.work_dir, 'ds')
        make_dir(cls.work_dir)

        with VVTKDataset(cls.path, mode='wb', num_shards=4,
                         compression=['none', 'none']) as ds:
            for i in range(cls.N):
                data = np.full((16,), float(i), dtype=np.float32)
                label = np.array([i], dtype=np.int64)
                ds.add(i, data, label)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_shuffle_covers_all(self):
        ds = VVTKDataset(self.path, mode='rb', compression=['none', 'none'])
        loader = VVTKDataLoader(ds, batch_size=self.batch_size, num_workers=2,
                                ring_size=4,
                                shapes=[(16,), (1,)],
                                dtypes=[torch.float32, torch.int64],
                                shuffle=True)
        all_labels = []
        for batch in loader:
            (_, _), (label, _) = batch
            all_labels.append(label.flatten())
            dot()
        all_labels = torch.cat(all_labels)
        unique = all_labels.unique().sort().values
        # All original indices must be present
        self.assertTrue(torch.equal(unique, torch.arange(self.N)))
        ds.close()


# ── 12. Edge cases ───────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):
    """Single sample, batch > dataset, exact-batch-size dataset."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_edge'
        make_dir(cls.work_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def _write_ds(self, name, N):
        subdir = os.path.join(self.work_dir, name)
        os.makedirs(subdir, exist_ok=True)
        path = os.path.join(subdir, 'data')
        write_dataset(path, 1, ['none', 'none'],
                      lambda i: (np.full((8,), float(i), dtype=np.float32),
                                 np.array([i], dtype=np.int64)), N)
        return path

    def test_single_sample_dataset_access(self):
        path = self._write_ds('single', 1)
        ds = VVTKDataset(path, mode='rb', compression=['none', 'none'])
        self.assertEqual(len(ds), 1)
        data, label = ds[0]
        self.assertTrue(torch.all(data == 0.0))
        self.assertEqual(label.item(), 0)
        dot()
        ds.close()

    def test_single_sample_torch_loader(self):
        path = self._write_ds('single_loader', 1)
        ds = VVTKDataset(path, mode='rb', compression=['none', 'none'])
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        seen = 0
        for batch in loader:
            data, label = batch
            seen += data.shape[0]
            dot()
        self.assertEqual(seen, 1)
        ds.close()

    def test_single_sample_cpp_loader(self):
        path = self._write_ds('single_cpp', 1)
        ds = VVTKDataset(path, mode='rb', compression=['none', 'none'])
        # Use batch_size=1 so the single sample forms a full batch
        loader = VVTKDataLoader(ds, batch_size=1, num_workers=1, ring_size=2,
                                shapes=[(8,), (1,)],
                                dtypes=[torch.float32, torch.int64])
        seen = 0
        for batch in loader:
            (data, _), (label, _) = batch
            seen += data.shape[0]
            dot()
        self.assertEqual(seen, 1)
        ds.close()

    def test_batch_larger_than_dataset(self):
        path = self._write_ds('small_ds', 5)
        ds = VVTKDataset(path, mode='rb', compression=['none', 'none'])
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        seen = 0
        for batch in loader:
            data, label = batch
            seen += data.shape[0]
            dot()
        self.assertEqual(seen, 5)
        ds.close()

    def test_exact_batch_size_dataset(self):
        """N is an exact multiple of batch_size — no partial batch."""
        path = self._write_ds('exact', 32)
        ds = VVTKDataset(path, mode='rb', compression=['none', 'none'])
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        batch_count = 0
        seen = 0
        for batch in loader:
            data, label = batch
            self.assertEqual(data.shape[0], 16)
            seen += data.shape[0]
            batch_count += 1
            dot()
        self.assertEqual(batch_count, 2)
        self.assertEqual(seen, 32)
        ds.close()


# ── 13. Torch tensors as input (not numpy) ──────────────────────────────────

class TestTorchTensorInput(unittest.TestCase):
    """Verify that torch.Tensor can be passed to ds.add() directly."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_torch_input'
        make_dir(cls.work_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_torch_tensor_write_read(self):
        N = 30
        path = os.path.join(self.work_dir, 'torch_in')
        with VVTKDataset(path, mode='wb', num_shards=2,
                         compression=['none', 'none']) as ds:
            for i in range(N):
                data = torch.full((3, 4, 4), float(i), dtype=torch.float32)
                label = torch.tensor([i], dtype=torch.int64)
                ds.add(i, data, label)

        ds = VVTKDataset(path, mode='rb', compression=['none', 'none'])
        for i in range(N):
            data, label = ds[i]
            self.assertTrue(torch.all(data == float(i)))
            self.assertEqual(label.item(), i)
            dot()
        ds.close()


# ── 14. Large sample count ───────────────────────────────────────────────────

class TestLargeSampleCount(unittest.TestCase):
    """Stress test with many small samples across many shards."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_large'
        cls.N = 5000
        cls.path = os.path.join(cls.work_dir, 'ds')
        make_dir(cls.work_dir)

        with VVTKDataset(cls.path, mode='wb', num_shards=16,
                         compression=['none']) as ds:
            for i in range(cls.N):
                t = np.array([i], dtype=np.int64)
                ds.add(i, t)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_random_access_all(self):
        ds = VVTKDataset(self.path, mode='rb', compression=['none'])
        self.assertEqual(len(ds), self.N)
        # Check every 100th + first/last
        indices = list(range(0, self.N, 100)) + [self.N - 1]
        for i in indices:
            (val,) = ds[i]
            self.assertEqual(val.item(), i)
            dot()
        ds.close()

    def test_torch_loader_full(self):
        ds = VVTKDataset(self.path, mode='rb', compression=['none'])
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
        seen = 0
        for batch in loader:
            (vals,) = batch
            seen += vals.shape[0]
            dot()
        self.assertEqual(seen, self.N)
        ds.close()

    def test_cpp_loader_full(self):
        ds = VVTKDataset(self.path, mode='rb', compression=['none'])
        loader = VVTKDataLoader(ds, batch_size=128, num_workers=4, ring_size=4,
                                shapes=[(1,)], dtypes=[torch.int64])
        real = 0
        for batch in loader:
            (data, lens), = batch
            for b in range(data.shape[0]):
                if lens[b].item() > 0:
                    real += 1
            dot()
        # C++ loader always drops last batch if not full
        expected = (self.N // 128) * 128
        self.assertEqual(real, expected)
        ds.close()


# ── 15. Two-epoch iteration ─────────────────────────────────────────────────

class TestTwoEpochs(unittest.TestCase):
    """Iterate the same loader twice (reset) and get consistent results."""

    @classmethod
    def setUpClass(cls):
        cls.work_dir = 'tests_temp_epochs'
        cls.N = 48
        cls.batch_size = 16
        cls.path = os.path.join(cls.work_dir, 'ds')
        make_dir(cls.work_dir)

        with VVTKDataset(cls.path, mode='wb', num_shards=2,
                         compression=['none', 'none']) as ds:
            for i in range(cls.N):
                data = np.full((4,), float(i), dtype=np.float32)
                label = np.array([i], dtype=np.int64)
                ds.add(i, data, label)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_cpp_loader_two_epochs(self):
        ds = VVTKDataset(self.path, mode='rb', compression=['none', 'none'])
        loader = VVTKDataLoader(ds, batch_size=self.batch_size, num_workers=2,
                                ring_size=4,
                                shapes=[(4,), (1,)],
                                dtypes=[torch.float32, torch.int64])
        for epoch in range(2):
            seen = 0
            for batch in loader:
                (data, _), (label, _) = batch
                seen += data.shape[0]
                dot()
            expected = ((self.N + self.batch_size - 1) // self.batch_size) * self.batch_size
            self.assertEqual(seen, expected)
        ds.close()

    def test_torch_loader_two_epochs(self):
        ds = VVTKDataset(self.path, mode='rb', compression=['none', 'none'])
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        for epoch in range(2):
            seen = 0
            for batch in loader:
                data, label = batch
                seen += data.shape[0]
                dot()
            self.assertEqual(seen, self.N)
        ds.close()


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Run with dot-only output
    unittest.main(verbosity=0)
