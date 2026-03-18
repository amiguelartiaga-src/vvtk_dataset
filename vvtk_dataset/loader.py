import torch
import numpy as np
from .base import VVTKBase
import _vvtk_core as core_lib
import time

class VVTKDataLoader:
    def __init__(self, dataset: VVTKBase, batch_size=32, num_workers=4, ring_size=4, 
                 shapes=None, dtypes=None, padding_values=None, shuffle=False,
                 nb_samples_per_epoch=None):
        """
        High-Performance C++ DataLoader with Decompression and Padding.
        
        Always drops the last batch when the dataset size is not evenly
        divisible by batch_size (the C++ loader pads the last batch to full
        size with zeros, so it is always discarded).
        
        Args:
            dataset: VVTKDataset instance.
            batch_size: Batch size.
            shapes (list of tuples): Max shape for each item in the tuple.
                                     Example: [(16000,), (50,)]
            dtypes (list of torch.dtype): Target dtype for each item.
                                          Example: [torch.float32, torch.int64]
            padding_values (list of float): Padding value for each item.
                                            Example: [0.0, -1.0]
            nb_samples_per_epoch (int): If set, each iteration yields at most
                this many samples (rounded down to full batches). The full
                dataset is consumed across consecutive mini-epochs and only
                reshuffled once all samples have been seen.
        """
        if not isinstance(dataset, VVTKBase): 
            raise TypeError("VVTKDataLoader requires a VVTKDataset object.")
        if shapes is None or dtypes is None: 
            raise ValueError("shapes and dtypes arguments are required.")
        if len(shapes) != len(dtypes):
            raise ValueError("shapes and dtypes must have the same length.")

        # Validate compression modes — only 'none', 'zstd', and 'flac' are
        # supported by the C++ loader.
        if hasattr(dataset, 'compression'):
            for mode in dataset.compression:
                if mode not in ('none', 'zstd', 'flac'):
                    raise ValueError(
                        f"VVTKDataLoader does not support compression='{mode}'. "
                        f"Only 'none', 'zstd', and 'flac' are supported by the C++ loader. "
                        f"Use a standard torch DataLoader for '{mode}' data."
                    )

        self.dataset = dataset
        self.batch_size = batch_size
        
        if padding_values is None:
            padding_values = [0.0] * len(shapes)

        # ---------------------------------------------------------------------
        # 1. Map Configuration to C++ Codes
        # ---------------------------------------------------------------------
        # Dtype Mapping (Matches C++ get_torch_dtype)
        dtype_map = {
            torch.uint8: 0, torch.int8: 1, torch.int16: 2,
            torch.int32: 3, torch.int64: 4,
            torch.float16: 10, torch.float32: 11, torch.float64: 12
        }
        cpp_dtypes = [dtype_map[d] for d in dtypes]

        # Compression mode mapping (matches C++ CompMode enum)
        comp_mode_map = {'none': 0, 'zstd': 1, 'flac': 2}
        if hasattr(dataset, 'compression'):
            cpp_comp_modes = [comp_mode_map[m] for m in dataset.compression]
        else:
            cpp_comp_modes = [0] * len(shapes)  # default: uncompressed
        
        # ---------------------------------------------------------------------
        # 2. Prepare Readers (Shards)
        # ---------------------------------------------------------------------
        # We pass a flat list of Reader pointers to C++.
        # We need a map to know which Reader corresponds to which Shard ID.
        shard_ids = sorted(dataset.readers.keys())
        cpp_readers = []
        shard_id_to_vec_idx = {}
        
        for i, sid in enumerate(shard_ids):
            cpp_readers.append(dataset.readers[sid])
            shard_id_to_vec_idx[sid] = i
            
        # ---------------------------------------------------------------------
        # 3. Build Global Index Map
        # ---------------------------------------------------------------------
        # This maps the logic index (0..N) to (Shard_Index, Local_File_Index)
        print("[VVTKLoader] Building index map...")
        t0 = time.time()
        
        max_idx = max(dataset.global_map.keys())
        global_map_vec = []
        
        # We assume indices are dense 0..N. If sparse, we fill gaps with dummy 0,0.
        for k in range(max_idx + 1):
            if k in dataset.global_map:
                shard_id = dataset.global_map[k]
                vec_idx = shard_id_to_vec_idx[shard_id]
                global_map_vec.append((vec_idx, k))
            else:
                global_map_vec.append((0, 0)) 

        print(f"[VVTKLoader] Ready in {time.time()-t0:.2f}s")

        # ---------------------------------------------------------------------
        # 4. Initialize C++ Core
        # ---------------------------------------------------------------------
        self.core = core_lib.VVTKLoader(
            cpp_readers,
            global_map_vec,
            [list(s) for s in shapes],
            cpp_dtypes,
            [float(p) for p in padding_values],
            cpp_comp_modes,
            batch_size, 
            num_workers, 
            ring_size,
            shuffle
        )
        # Always drop the last batch if not full (C++ pads with zeros)
        n_samples = len(dataset)
        self._full_length = n_samples // batch_size

        # Mini-epoch support
        if nb_samples_per_epoch is not None:
            self._epoch_length = min(nb_samples_per_epoch // batch_size, self._full_length)
            self._mini_epoch = True
        else:
            self._epoch_length = self._full_length
            self._mini_epoch = False
        self.length = self._epoch_length

        # Start "exhausted" so the first __iter__ triggers a shuffle/reset
        self._global_batch_pos = self._full_length

    def __iter__(self):
        # In mini-epoch mode, only reshuffle when the full dataset has been consumed.
        # In standard mode, always reshuffle at the start of every epoch.
        if not self._mini_epoch or self._global_batch_pos >= self._full_length:
            self.core.reset()
            self._global_batch_pos = 0
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self._epoch_length:
            raise StopIteration
        # Full dataset exhausted mid-epoch → short final mini-epoch
        if self._global_batch_pos >= self._full_length:
            raise StopIteration
        
        # Returns List[Tuple(Tensor, Tensor)] -> [(Data, Len), (Data, Len)...]
        results = self.core.next()
        
        if not results:
            raise StopIteration
            
        self.current_idx += 1
        self._global_batch_pos += 1
        # Clone tensors so callers can safely accumulate across iterations
        # (the C++ ring buffer may overwrite the underlying memory).
        return [(data.clone(), length.clone()) for data, length in results]

    def set_mini_epoch(self, mini_epoch):
        """Position the loader at the start of the given mini-epoch.

        Useful for resuming training from a checkpoint: restore the model
        and optimiser, then call ``loader.set_mini_epoch(n)`` so the next
        ``for batch in loader`` continues exactly where training left off.

        The method resets the C++ core and fast-forwards through the
        required number of batches.  For the **no-shuffle** (sequential)
        case the resulting data order is identical to an uninterrupted run.

        Note: with ``shuffle=True`` each ``reset()`` generates a new random
        permutation, so exact reproducibility requires external seed control.
        """
        if not self._mini_epoch:
            raise RuntimeError(
                "set_mini_epoch() requires nb_samples_per_epoch to be set."
            )

        total_batches = mini_epoch * self._epoch_length
        full_passes = total_batches // self._full_length
        remaining   = total_batches % self._full_length

        # First reset — beginning of pass 0
        self.core.reset()
        # Fast-forward through complete passes
        for _ in range(full_passes):
            for _ in range(self._full_length):
                self.core.next()
            self.core.reset()

        # Fast-forward within the current pass
        for _ in range(remaining):
            self.core.next()

        self._global_batch_pos = remaining

    def __len__(self):
        return self.length