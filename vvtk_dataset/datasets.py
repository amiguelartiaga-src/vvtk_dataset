import torch
import torch.nn.functional as F
import numpy as np
import io
from .base import VVTKBase

try:
    import _vvtk_core as core_lib
except ImportError:
    core_lib = None

# Compression mode mapping (matches C++ CompMode enum)
_COMP_MODE_MAP = {'none': 0, 'zstd': 1, 'flac': 2}

class VVTKDataset(VVTKBase):
    """
    Unified Zero-Copy Dataset for tuples of N tensors.
    Supports mixed compression modes and automatic padding on read.
    """
    def __init__(self, path, mode='rb', num_shards=32, compression=None, compression_args=None,
                 fixed_shapes=None, padding_values=None):
        """
        Args:
            path (str): File path prefix.
            mode (str): 'rb' or 'wb'.
            num_shards (int): Number of file shards for writing.
            compression (list of str): List of compression modes per item (e.g., ['flac', 'none']).
            compression_args (list of dict): Optional args.
            fixed_shapes (list of tuple): Optional. If set, __getitem__ pads/truncates tensors
                                          to this shape and returns (tensor, length).
            padding_values (list of float): Optional. Values to pad with if fixed_shapes is set.
        """
        super().__init__(path, mode, num_shards)
        
        # Default configuration
        if compression is None:
            self.compression = ['none']
        else:
            self.compression = compression
            
        if compression_args is None:
            self.compression_args = [{}] * len(self.compression)
        else:
            self.compression_args = compression_args
            
        # Padding Configuration
        self.fixed_shapes = fixed_shapes
        self.padding_values = padding_values
        
        if self.fixed_shapes is not None:
            if self.padding_values is None:
                self.padding_values = [0.0] * len(self.fixed_shapes)
            if len(self.fixed_shapes) != len(self.compression):
                raise ValueError("fixed_shapes length must match compression/items length")

        # Validation
        valid_modes = {'none', 'zstd', 'flac'}
        for c in self.compression:
            if c not in valid_modes:
                raise ValueError(f"Unknown compression mode: {c}. Valid: {valid_modes}")

    def add(self, key, *tensors):
        """
        Write a sample (tuple of N tensors) to the dataset.
        Requires 'zstd' and/or 'soundfile' Python packages for compressed modes.
        """
        if 'w' not in self.mode: 
            raise RuntimeError("Not in write mode")
        
        if len(tensors) != len(self.compression):
            raise ValueError(f"Expected {len(self.compression)} tensors, got {len(tensors)}")
        
        blob = bytearray()
        
        for i, tensor in enumerate(tensors):
            if isinstance(tensor, torch.Tensor): 
                tensor = tensor.numpy()
            
            # Ensure contiguous memory for safe C++ consumption
            tensor = np.ascontiguousarray(tensor)
            
            # 1. Generate Header (64 bytes)
            header = self._make_header(tensor)
            blob.extend(header)
            
            # 2. Compress Payload
            mode = self.compression[i]
            args = self.compression_args[i]
            
            if mode == 'none':
                blob.extend(tensor.tobytes())
            elif mode == 'zstd':
                import zstd
                level = args.get('level', 3)
                data = zstd.compress(tensor.tobytes(), level)
                blob.extend(np.array([len(data)], dtype=np.int64).tobytes())
                blob.extend(data)
            elif mode == 'flac':
                import soundfile as sf
                sr = args.get('sample_rate', 16000)
                with io.BytesIO() as buf:
                    sf.write(buf, tensor, sr, format='FLAC', subtype='PCM_16')
                    data = buf.getvalue()
                blob.extend(np.array([len(data)], dtype=np.int64).tobytes())
                blob.extend(data)
            
            # 3. Alignment Padding
            curr_len = len(blob)
            if curr_len % 8 != 0:
                blob.extend(b'\x00' * (8 - (curr_len % 8)))
                
        self._write_blob(key, blob)

    def __getitem__(self, index):
        """
        Reads a sample using the C++ decompression backend.
        Uncompressed items are returned as zero-copy views into the mmap'd file.
        Compressed items (zstd, flac) are decompressed in C++.
        
        If fixed_shapes is None: returns (Tensor1, Tensor2, ...)
        If fixed_shapes is Set:  returns ((Tensor1, Len1), (Tensor2, Len2), ...)
        """
        blob = self._get_blob(index)
        
        # Map compression strings to C++ enum codes
        comp_modes = [_COMP_MODE_MAP[m] for m in self.compression]
        
        # C++ decodes all items at once: zero-copy for 'none', decompresses for zstd/flac
        tensors = core_lib.decode_blob_items(blob, comp_modes)
        
        results = []
        for i, tensor in enumerate(tensors):
            # Automatic Padding (Python Side)
            if self.fixed_shapes is not None:
                target_shape = self.fixed_shapes[i]
                pad_val = self.padding_values[i]
                real_len = tensor.shape[0]  # Assuming dim 0 is the variable length
                max_len = target_shape[0]
                
                # Clone needed: zero-copy views can't be resized,
                # and we need owned memory for padding
                if real_len > max_len:
                    tensor = tensor[:max_len].clone()
                    real_len = max_len
                elif real_len < max_len:
                    tensor = tensor.clone()
                    pad_shape = list(tensor.shape)
                    pad_shape[0] = max_len - real_len
                    padding = torch.full(pad_shape, pad_val, dtype=tensor.dtype)
                    tensor = torch.cat([tensor, padding], dim=0)
                else:
                    # Exact match — clone only if it's a view (uncompressed zero-copy)
                    if comp_modes[i] == 0:
                        tensor = tensor.clone()
                
                results.append((tensor, torch.tensor(real_len, dtype=torch.int64)))
            else:
                results.append(tensor)
        
        return tuple(results)