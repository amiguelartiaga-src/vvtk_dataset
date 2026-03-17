import os
import pickle
import glob
import numpy as np
import torch

try:
    import _vvtk_core as core_lib
except ImportError:
    core_lib = None

class VVTKBase:
    def __init__(self, path, mode='rb', num_shards=32):
        self.path = path
        self.mode = mode
        self.num_shards = num_shards
        
        self.dtype2code = {
            np.dtype('uint8'): 0, np.dtype('int8'): 1, np.dtype('int16'): 2,
            np.dtype('int32'): 3, np.dtype('int64'): 4,
            np.dtype('float16'): 10, np.dtype('float32'): 11, np.dtype('float64'): 12
        }
        self.code2dtype = {
            0: torch.uint8, 1: torch.int8, 2: torch.int16, 3: torch.int32, 4: torch.int64,
            10: torch.float16, 11: torch.float32, 12: torch.float64
        }

        self.shards = []
        self.shard_indices = []
        self.global_map = {}
        self.readers = {} # Dictionary: {shard_id: VVTKReader_Object}
        self.lookup = None

        if 'w' in mode:
            self._init_writer()
        else:
            self._init_reader()

    # ... (Keep existing write methods: __enter__, __exit__, close, _init_writer, _write_blob, _make_header) ...
    # Copy them from your original code or the previous prompt's context

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    
    def close(self):
        if 'w' in self.mode and self.shards:
            for i, f in enumerate(self.shards):
                pos = f.tell()
                pickle.dump(self.shard_indices[i], f)
                f.seek(0)
                f.write(pos.to_bytes(8, 'little', signed=True))
                f.close()
            with open(f"{self.path}.master", 'wb') as f:
                pickle.dump(self.global_map, f)
            self.shards = []
            print(f"[VVTK] Saved dataset to {self.path}")
        self.readers = {}

    def _init_writer(self):
        base_dir = os.path.dirname(self.path)
        if base_dir: os.makedirs(base_dir, exist_ok=True)
        self.shard_indices = [{} for _ in range(self.num_shards)]
        for i in range(self.num_shards):
            fname = f"{self.path}.{i:03d}.vvtk"
            f = open(fname, 'wb')
            f.write((0).to_bytes(8, 'little', signed=True))
            self.shards.append(f)

    def _write_blob(self, key, blob):
        if 'w' not in self.mode: raise RuntimeError("Not in write mode")
        shard_id = key % self.num_shards
        f = self.shards[shard_id]
        pos = f.tell()
        if pos % 8 != 0:
            pad = 8 - (pos % 8)
            f.write(b'\x00' * pad)
            pos += pad
        f.write(blob)
        self.shard_indices[shard_id][key] = (pos, len(blob))
        self.global_map[key] = shard_id

    def _make_header(self, tensor):
        if isinstance(tensor, torch.Tensor): tensor = tensor.numpy()
        header = np.zeros(8, dtype=np.int64)
        header[0] = self.dtype2code[tensor.dtype]
        header[1] = tensor.ndim
        for i, s in enumerate(tensor.shape): header[2+i] = s
        return header.tobytes()

    def _init_reader(self):
        if core_lib is None: raise ImportError("C++ Extension not found.")
        master_path = f"{self.path}.master"
        if not os.path.exists(master_path): raise FileNotFoundError(f"Index {master_path} not found")

        with open(master_path, 'rb') as f:
            global_map = pickle.load(f)
        
        if not global_map: raise ValueError("Empty dataset")
        self.global_map = global_map
        
        max_key = max(global_map.keys())
        self.lookup = np.full(max_key + 1, -1, dtype=np.int32)
        for k, s_id in global_map.items():
            self.lookup[k] = s_id

        shard_files = sorted(glob.glob(f"{self.path}.*.vvtk"))
        for fname in shard_files:
            try: shard_id = int(fname.split('.')[-2])
            except: continue
            with open(fname, 'rb') as f:
                pos = int.from_bytes(f.read(8), 'little', signed=True)
                f.seek(pos)
                raw_index = pickle.load(f)
            if not raw_index: continue
            
            k_max = max(raw_index.keys())
            offsets = np.zeros(k_max + 1, dtype=np.int64)
            lengths = np.zeros(k_max + 1, dtype=np.int64)
            for k, (off, length) in raw_index.items():
                offsets[k] = off
                lengths[k] = length
            
            # Create C++ Reader
            self.readers[shard_id] = core_lib.VVTKReader(fname, offsets.tolist(), lengths.tolist())

    def _get_blob(self, index):
        if 'r' not in self.mode: raise RuntimeError("Not in read mode")
        if index >= len(self.lookup): raise IndexError("Index out of bounds")
        shard_id = self.lookup[index]
        if shard_id == -1: raise ValueError(f"Missing key {index}")
        return self.readers[shard_id].get_blob_view(index)

    def __len__(self):
        return len(self.global_map) if self.global_map else 0

