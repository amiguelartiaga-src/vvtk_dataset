import unittest
import os
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from vvtk_dataset import VVTKDataset, VVTKDataLoader

# Helpers for deterministic variable length generation
def get_sample_info(idx, min_l1=100, max_l1=32000, min_l2=10, max_l2=512):
    rs = np.random.RandomState(seed=idx) 
    l1 = rs.randint(min_l1, max_l1 + 1)
    l2 = rs.randint(min_l2, max_l2 + 1)
    return l1, l2

class TestVariableLength(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Called once before all tests."""
        cls.work_dir = "tests_temp_var_len"
        cls.dataset_path = os.path.join(cls.work_dir, "var_len.vvtk")
        cls.num_samples = 1000
        cls.batch_size = 32
        
        # Max Shapes for padding
        cls.max_shape1 = (32000,)
        cls.max_shape2 = (512,)
        
        # Padding values
        cls.pad_val1 = 0.0
        cls.pad_val2 = -1
        
        if os.path.exists(cls.work_dir):
            shutil.rmtree(cls.work_dir)
        os.makedirs(cls.work_dir)
        
        # --- GENERATE DATASET ---
        print(f"\n[Setup] Generating {cls.num_samples} variable length samples...")
        with VVTKDataset(cls.dataset_path, mode='wb', num_shards=4, compression=['none', 'none']) as ds:
            for i in range(cls.num_samples):
                l1, l2 = get_sample_info(i)
                val = float(i % 2048) 
                t1 = np.full((l1,), fill_value=val, dtype=np.float16)
                t2 = np.full((l2,), fill_value=i, dtype=np.int32)
                ds.add(i, t1, t2)

    @classmethod
    def tearDownClass(cls):
        """Called once after all tests."""
        if os.path.exists(cls.work_dir):
            shutil.rmtree(cls.work_dir)

    def test_01_dataset_correctness(self):
        """Test individual random access retrieval (No Padding)."""
        print("\n--- Test 1: VVTKDataset Variable Length Raw Access ---")
        ds = VVTKDataset(self.dataset_path, mode='rb', compression=['none', 'none'])
        
        indices = [0, 10, 500, 999]
        for idx in indices:
            # When fixed_shapes is None, returns (t1, t2)
            t1, t2 = ds[idx]
            exp_l1, exp_l2 = get_sample_info(idx)
            
            self.assertEqual(t1.shape[0], exp_l1)
            self.assertEqual(t2.shape[0], exp_l2)
            
            val = float(idx % 2048)
            self.assertTrue(torch.all(t1 == val))
            self.assertTrue(torch.all(t2 == idx))
            
        ds.close()

    def test_02_pytorch_dataloader_internal_padding(self):
        """
        Test PyTorch DataLoader using VVTKDataset's internal padding logic.
        No custom collate_fn is needed anymore.
        """
        print("\n--- Test 2: Standard PyTorch DataLoader (Dataset-side Padding) ---")
        
        # Configure Dataset to pad internally
        ds = VVTKDataset(
            self.dataset_path, 
            mode='rb', 
            compression=['none', 'none'],
            fixed_shapes=[self.max_shape1, self.max_shape2],
            padding_values=[self.pad_val1, self.pad_val2]
        )
        
        # Standard DataLoader using default_collate
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        samples_seen = 0
        for i, batch in enumerate(loader):
            # Batch structure matches C++ loader: ((Data1, Len1), (Data2, Len2))
            (d1, l1), (d2, l2) = batch
            
            curr_bs = d1.shape[0]
            start_idx = i * self.batch_size
            
            # Verify batch structure
            self.assertEqual(d1.shape, (curr_bs, 32000))
            self.assertEqual(d2.shape, (curr_bs, 512))
            
            # Verify Content
            for b in range(curr_bs):
                global_idx = start_idx + b
                exp_l1, exp_l2 = get_sample_info(global_idx)
                
                # Check lengths
                self.assertEqual(l1[b].item(), exp_l1)
                self.assertEqual(l2[b].item(), exp_l2)
                
                # Check Valid Data
                val = float(global_idx % 2048)
                self.assertTrue(torch.all(d1[b, :exp_l1] == val))
                self.assertTrue(torch.all(d2[b, :exp_l2] == global_idx))
                
                # Check Padding
                if exp_l1 < 32000:
                    self.assertTrue(torch.all(d1[b, exp_l1:] == self.pad_val1))
                if exp_l2 < 512:
                    self.assertTrue(torch.all(d2[b, exp_l2:] == int(self.pad_val2)))
                    
            samples_seen += curr_bs
            
        self.assertEqual(samples_seen, self.num_samples)
        ds.close()

    def test_03_vvtk_cpp_dataloader(self):
        """Test the C++ VVTKDataLoader."""
        print("\n--- Test 3: C++ VVTKDataLoader (Auto Padding) ---")
        ds = VVTKDataset(self.dataset_path, mode='rb', compression=['none', 'none'])
        
        loader = VVTKDataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=4,
            ring_size=4,
            shapes=[self.max_shape1, self.max_shape2],
            dtypes=[torch.float16, torch.int32],
            padding_values=[self.pad_val1, self.pad_val2]
        )
        
        samples_seen = 0
        
        for batch_tuple in loader:
            (d1, l1), (d2, l2) = batch_tuple
            current_bs = d1.shape[0]
            
            self.assertEqual(d1.shape[1:], self.max_shape1)
            self.assertEqual(d2.shape[1:], self.max_shape2)
            
            for b in range(current_bs):
                real_len1 = l1[b].item()
                real_len2 = l2[b].item()
                
                # Skip end-of-epoch loader padding
                if real_len1 == 0 and real_len2 == 0:
                    continue

                global_idx = d2[b, 0].item()
                exp_l1, exp_l2 = get_sample_info(global_idx)
                
                self.assertEqual(real_len1, exp_l1)
                self.assertEqual(real_len2, exp_l2)
                
                val_float = float(global_idx % 2048)
                
                # Check Data
                if not torch.all(d1[b, :real_len1] == val_float):
                    self.fail(f"Float16 Data mismatch at idx {global_idx}")
                if not torch.all(d2[b, :real_len2] == global_idx):
                    self.fail(f"Int32 Data mismatch at idx {global_idx}")
                
                # Check Padding
                if real_len1 < 32000:
                    if not torch.all(d1[b, real_len1:] == self.pad_val1):
                        self.fail(f"Float16 Padding corrupt at idx {global_idx}")
                if real_len2 < 512:
                     if not torch.all(d2[b, real_len2:] == int(self.pad_val2)):
                         self.fail(f"Int32 Padding corrupt at idx {global_idx}")

                samples_seen += 1

        print(f"Seen {samples_seen} real samples (expected {(self.num_samples // self.batch_size) * self.batch_size}).")
        self.assertEqual(samples_seen, (self.num_samples // self.batch_size) * self.batch_size)
        ds.close()

if __name__ == '__main__':
    unittest.main()