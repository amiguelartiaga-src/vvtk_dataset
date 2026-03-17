import unittest
import os
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from vvtk_dataset import VVTKDataset, VVTKDataLoader

# Helpers for deterministic variable-length sinusoid generation
def get_sample_info(idx, min_len=4000, max_len=32000, sr=16000):
    rs = np.random.RandomState(seed=idx)
    length = rs.randint(min_len, max_len + 1)
    freq = 200.0 + (idx % 50) * 80.0          # 200-4120 Hz
    return length, freq

def generate_sinusoid(idx, sr=16000):
    """Generate a deterministic sinusoid for sample *idx*."""
    length, freq = get_sample_info(idx, sr=sr)
    t = np.arange(length, dtype=np.float32) / sr
    waveform = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return waveform


class TestFlacCompression(unittest.TestCase):
    """End-to-end tests for FLAC compression through the C++ VVTKDataLoader."""

    SR = 16000

    @classmethod
    def setUpClass(cls):
        cls.work_dir = "tests_temp_flac"
        cls.dataset_path = os.path.join(cls.work_dir, "flac_test.vvtk")
        cls.num_samples = 500
        cls.batch_size = 32
        cls.max_audio = (32000,)
        cls.label_shape = (1,)
        cls.pad_val_audio = 0.0
        cls.pad_val_label = 0

        if os.path.exists(cls.work_dir):
            shutil.rmtree(cls.work_dir)
        os.makedirs(cls.work_dir)

        print(f"\n[Setup] Generating {cls.num_samples} FLAC-compressed samples...")
        with VVTKDataset(
            cls.dataset_path, mode='wb', num_shards=4,
            compression=['flac', 'none'],
            compression_args=[{'sample_rate': cls.SR}, {}]
        ) as ds:
            for i in range(cls.num_samples):
                waveform = generate_sinusoid(i, cls.SR)
                label = np.array([i], dtype=np.int64)
                ds.add(i, waveform, label)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.work_dir):
            shutil.rmtree(cls.work_dir)

    # ------------------------------------------------------------------
    # Test 1: Python-side raw access (soundfile decode)
    # ------------------------------------------------------------------
    def test_01_python_raw_access(self):
        """Read FLAC samples via Python __getitem__ (soundfile decode)."""
        print("\n--- Test 1: VVTKDataset FLAC Raw Access (Python) ---")
        ds = VVTKDataset(
            self.dataset_path, mode='rb',
            compression=['flac', 'none'],
            compression_args=[{'sample_rate': self.SR}, {}]
        )

        for idx in [0, 10, 250, self.num_samples - 1]:
            audio, label = ds[idx]
            expected_len, freq = get_sample_info(idx, sr=self.SR)

            # FLAC is lossy at PCM_16 subtype → allow some tolerance
            self.assertEqual(label.item(), idx)
            # Length should match (FLAC is lossless in sample count)
            self.assertEqual(audio.shape[0], expected_len,
                             f"Length mismatch at idx {idx}")

            # Verify sinusoid shape: check that the signal energy is in
            # the right ballpark (not silence, not garbage)
            rms = torch.sqrt(torch.mean(audio ** 2)).item()
            self.assertGreater(rms, 0.1,
                               f"RMS too low ({rms}) at idx {idx} — likely silence/garbage")
            self.assertLess(rms, 1.0,
                            f"RMS too high ({rms}) at idx {idx}")

        ds.close()

    # ------------------------------------------------------------------
    # Test 2: PyTorch DataLoader with dataset-side padding
    # ------------------------------------------------------------------
    def test_02_pytorch_dataloader(self):
        """FLAC decode via PyTorch DataLoader with internal padding."""
        print("\n--- Test 2: FLAC + PyTorch DataLoader (dataset-side padding) ---")
        ds = VVTKDataset(
            self.dataset_path, mode='rb',
            compression=['flac', 'none'],
            compression_args=[{'sample_rate': self.SR}, {}],
            fixed_shapes=[self.max_audio, self.label_shape],
            padding_values=[self.pad_val_audio, self.pad_val_label]
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=2)

        samples_seen = 0
        for i, batch in enumerate(loader):
            (d_audio, l_audio), (d_label, l_label) = batch
            curr_bs = d_audio.shape[0]
            self.assertEqual(d_audio.shape[1], self.max_audio[0])

            for b in range(curr_bs):
                global_idx = i * self.batch_size + b
                exp_len, _ = get_sample_info(global_idx, sr=self.SR)
                self.assertEqual(l_audio[b].item(), exp_len)
                self.assertEqual(d_label[b].item(), global_idx)

                # Check padding region is zero
                if exp_len < self.max_audio[0]:
                    pad_region = d_audio[b, exp_len:]
                    self.assertTrue(torch.all(pad_region == 0),
                                    f"Padding not zero at idx {global_idx}")

            samples_seen += curr_bs

        self.assertEqual(samples_seen, self.num_samples)
        ds.close()

    # ------------------------------------------------------------------
    # Test 3: C++ VVTKDataLoader with FLAC decode
    # ------------------------------------------------------------------
    def test_03_cpp_dataloader(self):
        """FLAC decode in C++ VVTKDataLoader (dr_flac)."""
        print("\n--- Test 3: FLAC + C++ VVTKDataLoader ---")
        ds = VVTKDataset(
            self.dataset_path, mode='rb',
            compression=['flac', 'none'],
            compression_args=[{'sample_rate': self.SR}, {}]
        )

        loader = VVTKDataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=4,
            ring_size=4,
            shapes=[self.max_audio, self.label_shape],
            dtypes=[torch.float32, torch.int64],
            padding_values=[self.pad_val_audio, self.pad_val_label],
        )

        samples_seen = 0
        for batch_tuple in loader:
            (d_audio, l_audio), (d_label, l_label) = batch_tuple
            self.assertEqual(d_audio.shape[1], self.max_audio[0])

            for b in range(d_audio.shape[0]):
                real_len = l_audio[b].item()
                if real_len == 0:
                    continue  # skip end-of-epoch padding

                label_val = d_label[b].item()
                exp_len, freq = get_sample_info(label_val, sr=self.SR)
                self.assertEqual(real_len, exp_len,
                                 f"Length mismatch at label {label_val}: "
                                 f"got {real_len}, expected {exp_len}")

                # Verify audio data is non-trivial
                audio_data = d_audio[b, :real_len]
                rms = torch.sqrt(torch.mean(audio_data ** 2)).item()
                self.assertGreater(rms, 0.1,
                                   f"RMS too low ({rms}) at label {label_val}")

                # Verify padding is zero
                if real_len < self.max_audio[0]:
                    pad_region = d_audio[b, real_len:]
                    self.assertTrue(torch.all(pad_region == 0),
                                    f"Padding not zero at label {label_val}")

                samples_seen += 1

        expected = (self.num_samples // self.batch_size) * self.batch_size
        print(f"Seen {samples_seen} real samples (expected {expected}).")
        self.assertEqual(samples_seen, expected)
        ds.close()

    # ------------------------------------------------------------------
    # Test 4: Cross-check Python vs C++ decode
    # ------------------------------------------------------------------
    def test_04_python_vs_cpp_equivalence(self):
        """Verify C++ dr_flac decode matches Python soundfile decode."""
        print("\n--- Test 4: Python vs C++ FLAC decode equivalence ---")

        # Python side: raw access
        ds_py = VVTKDataset(
            self.dataset_path, mode='rb',
            compression=['flac', 'none'],
            compression_args=[{'sample_rate': self.SR}, {}]
        )

        # C++ side: single-sample batches for easy comparison
        ds_cpp = VVTKDataset(
            self.dataset_path, mode='rb',
            compression=['flac', 'none'],
            compression_args=[{'sample_rate': self.SR}, {}]
        )
        loader = VVTKDataLoader(
            ds_cpp,
            batch_size=1,
            num_workers=1,
            ring_size=2,
            shapes=[self.max_audio, self.label_shape],
            dtypes=[torch.float32, torch.int64],
            padding_values=[0.0, 0],
        )

        # Compare first N samples
        check_count = min(50, self.num_samples)
        cpp_iter = iter(loader)
        for idx in range(check_count):
            py_audio, py_label = ds_py[idx]
            py_len = py_audio.shape[0]

            batch = next(cpp_iter)
            (cpp_audio, cpp_len), (cpp_label, _) = batch
            cpp_real_len = cpp_len[0].item()

            self.assertEqual(py_len, cpp_real_len,
                             f"Length mismatch at idx {idx}: py={py_len}, cpp={cpp_real_len}")

            # Both decode from same FLAC bitstream; allow small float tolerance
            # (soundfile normalises int16→float32 the same way dr_flac does)
            py_data = py_audio[:py_len]
            cpp_data = cpp_audio[0, :cpp_real_len]
            max_diff = (py_data - cpp_data).abs().max().item()
            self.assertLess(max_diff, 1e-3,
                            f"Data mismatch at idx {idx}: max_diff={max_diff}")

        ds_py.close()
        ds_cpp.close()
        print(f"Checked {check_count} samples — Python and C++ decodes match.")


if __name__ == '__main__':
    unittest.main()
