import unittest
import os
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from vvtk_dataset import VVTKDataset, VVTKDataLoader

class TestFixedTensors(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Called once before all tests."""
        cls.work_dir = "tests_temp_data"
        cls.dataset_path = os.path.join(cls.work_dir, "fixed_test.vvtk")
        cls.num_samples = 1000
        cls.batch_size = 32
        
        # Data Config
        cls.data_shape = (3, 112, 112)
        cls.label_shape = (1,)
        
        if os.path.exists(cls.work_dir):
            shutil.rmtree(cls.work_dir)
        os.makedirs(cls.work_dir)
        
        # --- GENERATE DATASET ---
        print(f"\n[Setup] Generating {cls.num_samples} samples...")
        with VVTKDataset(cls.dataset_path, mode='wb', num_shards=4, compression=['none', 'none']) as ds:
            for i in range(cls.num_samples):
                # Deterministic data: fill tensor with value = i
                # Using float32 for data, int64 for label
                data = np.full(cls.data_shape, fill_value=float(i), dtype=np.float32)
                label = np.array([i], dtype=np.int64)
                ds.add(i, data, label)

    @classmethod
    def tearDownClass(cls):
        """Called once after all tests."""
        if os.path.exists(cls.work_dir):
            shutil.rmtree(cls.work_dir)

    def test_01_dataset_correctness(self):
        """Test individual random access retrieval."""
        print("\n--- Test 1: VVTKDataset Raw Access ---")
        ds = VVTKDataset(self.dataset_path, mode='rb', compression=['none', 'none'])
        
        indices_to_check = [0, self.num_samples // 2, self.num_samples - 1]
        
        for idx in indices_to_check:
            data, label = ds[idx]
            print(data.shape, data.dtype, label.shape, label.dtype)
            break
        
        for idx in indices_to_check:
            data, label = ds[idx]
            
            # Check Types
            self.assertTrue(torch.is_tensor(data))
            self.assertTrue(torch.is_tensor(label))
            
            # Check Shapes
            self.assertEqual(data.shape, self.data_shape)
            self.assertEqual(label.shape, self.label_shape)
            
            # Check Values (Exact Match)
            expected_val = float(idx)
            self.assertTrue(torch.all(data == expected_val), f"Data mismatch at idx {idx}")
            self.assertEqual(label.item(), idx, f"Label mismatch at idx {idx}")
            
        ds.close()

    def test_02_pytorch_dataloader(self):
        """Test compatibility with Standard PyTorch DataLoader."""
        print("\n--- Test 2: Standard PyTorch DataLoader ---")
        ds = VVTKDataset(self.dataset_path, mode='rb', compression=['none', 'none'])
        
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        for i, batch in enumerate(loader):
            data_batch, label_batch = batch
            print(data_batch.shape, data_batch.dtype, label_batch.shape, label_batch.dtype)
            break
             
        samples_seen = 0
        for i, batch in enumerate(loader):
            data_batch, label_batch = batch
            
            # Check Batch Sizes
            current_bs = data_batch.shape[0]
            self.assertEqual(data_batch.shape, (current_bs, *self.data_shape))
            self.assertEqual(label_batch.shape, (current_bs, *self.label_shape))
            
            # Check Values (Sequential)
            start_idx = i * self.batch_size
            expected_labels = torch.arange(start_idx, start_idx + current_bs, dtype=torch.int64).unsqueeze(1)
            
            self.assertTrue(torch.equal(label_batch, expected_labels), 
                            f"Batch {i} labels mismatch")
            
            # Check Data content (first item in batch)
            first_val = data_batch[0].mean().item()
            self.assertEqual(first_val, float(start_idx), f"Batch {i} data value mismatch")
            
            samples_seen += current_bs
            
        self.assertEqual(samples_seen, self.num_samples)
        ds.close()

    def test_03_vvtk_cpp_dataloader(self):
        """Test the C++ VVTKDataLoader."""
        print("\n--- Test 3: C++ VVTKDataLoader ---")
        ds = VVTKDataset(self.dataset_path, mode='rb', compression=['none', 'none'])
        
        loader = VVTKDataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=4,
            ring_size=4,
            shapes=[self.data_shape, self.label_shape],
            dtypes=[torch.float32, torch.int64]
        )
        
        for i, batch in enumerate(loader):
            data_batch, label_batch = batch
            print(data_batch[0].shape, data_batch[0].dtype, data_batch[1].shape, data_batch[1].dtype)
            print(label_batch[0].shape, label_batch[0].dtype, label_batch[1].shape, label_batch[1].dtype)
            break
        
        samples_seen = 0
                
        for i, batch_tuple in enumerate(loader):
            # Output format: [(Data, Len), (Label, Len)]
            (data_batch, data_lens), (label_batch, label_lens) = batch_tuple
            
            current_bs = data_batch.shape[0]
            
            # 1. Check Shapes
            self.assertEqual(data_batch.shape[1:], self.data_shape)
            self.assertEqual(label_batch.shape[1:], self.label_shape)
            
            # 2. Verify Integrity: Data value must match Label value
            # Data is [B, 3, 112, 112]. Extract a scalar pixel (all are equal to index).
            data_vals = data_batch[:, 0, 0, 0].long() 
            label_vals = label_batch.flatten()
            
            if not torch.equal(data_vals, label_vals):
                print(f"\n[FAILURE] Mismatch in batch {i} (Size {current_bs})")
                
                # Find the bad index
                mask = (data_vals != label_vals)
                bad_indices = mask.nonzero(as_tuple=True)[0]
                first_bad = bad_indices[0].item()
                
                val_d = data_vals[first_bad].item()
                val_l = label_vals[first_bad].item()
                
                print(f"Index {first_bad} in batch: Data={val_d}, Label={val_l}")
                print(f"Data Batch Slice: {data_vals[:10]}")
                print(f"Label Batch Slice: {label_vals[:10]}")
                
                self.fail(f"Data content ({val_d}) does not match Label content ({val_l})")
            
            samples_seen += current_bs

        # The C++ loader always drops the last batch if not full
        expected_samples = (self.num_samples // self.batch_size) * self.batch_size
        
        print(f"Seen {samples_seen} samples. Expected {expected_samples} (drop last).")
        self.assertEqual(samples_seen, expected_samples)
        ds.close()

if __name__ == '__main__':
    unittest.main()