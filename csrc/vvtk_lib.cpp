#include <torch/extension.h>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstring>
#include <memory>
#include <algorithm>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// ZSTD DECOMPRESSION (symbols provided by bundled zstddeclib.c)
// ============================================================================
extern "C" {
    size_t ZSTD_decompress(void* dst, size_t dstCapacity,
                          const void* src, size_t compressedSize);
    unsigned ZSTD_isError(size_t code);
    const char* ZSTD_getErrorName(size_t code);
}

// ============================================================================
// FLAC DECOMPRESSION (symbols provided by bundled dr_flac_impl.c)
// ============================================================================
#include "dr_flac.h"

// Compression mode codes (must match Python side)
enum CompMode : int { COMP_NONE = 0, COMP_ZSTD = 1, COMP_FLAC = 2 };

// ============================================================================
// HELPERS
// ============================================================================
inline size_t get_dtype_size(int64_t code) {
    switch(code) {
        case 0: return 1; // uint8
        case 1: return 1; // int8
        case 2: return 2; // int16
        case 3: return 4; // int32
        case 4: return 8; // int64
        case 10: return 2; // float16
        case 11: return 4; // float32
        case 12: return 8; // float64
        default: return 1; 
    }
}

inline torch::ScalarType get_torch_dtype(int64_t code) {
    switch(code) {
        case 0: return torch::kUInt8;
        case 1: return torch::kInt8;
        case 2: return torch::kInt16;
        case 3: return torch::kInt32;
        case 4: return torch::kInt64;
        case 10: return torch::kFloat16;
        case 11: return torch::kFloat32;
        case 12: return torch::kFloat64;
        default: return torch::kFloat32;
    }
}

// ============================================================================
// SHARED DECOMPRESSION HELPERS
// Used by both VVTKLoader (into pre-allocated ring buffers) and
// decode_blob_items (returns new tensors for dataset __getitem__).
// ============================================================================

// Decompress zstd data from src into dst. Returns decompressed size in bytes,
// or 0 on error.
inline size_t decompress_zstd_to(const uint8_t* src, size_t src_len,
                                 uint8_t* dst, size_t dst_capacity) {
    size_t res = ZSTD_decompress(dst, dst_capacity, src, src_len);
    if (ZSTD_isError(res)) return 0;
    return res;
}

// Decompress FLAC data into a float32 buffer. Returns number of frames decoded.
inline size_t decompress_flac_f32(const uint8_t* src, size_t src_len,
                                   float* dst, size_t max_frames) {
    drflac* pFlac = drflac_open_memory(src, src_len, NULL);
    if (!pFlac) return 0;
    size_t frames_to_read = std::min((size_t)pFlac->totalPCMFrameCount, max_frames);
    size_t frames_read = drflac_read_pcm_frames_f32(pFlac, frames_to_read, dst);
    drflac_close(pFlac);
    return frames_read;
}

// Decompress FLAC data into an int16 buffer. Returns number of frames decoded.
inline size_t decompress_flac_s16(const uint8_t* src, size_t src_len,
                                   int16_t* dst, size_t max_frames) {
    drflac* pFlac = drflac_open_memory(src, src_len, NULL);
    if (!pFlac) return 0;
    size_t frames_to_read = std::min((size_t)pFlac->totalPCMFrameCount, max_frames);
    size_t frames_read = drflac_read_pcm_frames_s16(pFlac, frames_to_read, (drflac_int16*)dst);
    drflac_close(pFlac);
    return frames_read;
}

// Decompress FLAC data into an int32 buffer. Returns number of frames decoded.
inline size_t decompress_flac_s32(const uint8_t* src, size_t src_len,
                                   int32_t* dst, size_t max_frames) {
    drflac* pFlac = drflac_open_memory(src, src_len, NULL);
    if (!pFlac) return 0;
    size_t frames_to_read = std::min((size_t)pFlac->totalPCMFrameCount, max_frames);
    size_t frames_read = drflac_read_pcm_frames_s32(pFlac, frames_to_read, (drflac_int32*)dst);
    drflac_close(pFlac);
    return frames_read;
}

// ============================================================================
// PART 1: VVTK READER
// ============================================================================
class VVTKReader {
private:
    int fd;
    size_t file_size;
    uint8_t* data_ptr;
public:
    std::vector<int64_t> offsets;
    std::vector<int64_t> lengths;

    VVTKReader(std::string filename, std::vector<int64_t> idx_offsets, std::vector<int64_t> idx_lengths) {
        fd = open(filename.c_str(), O_RDONLY);
        if (fd == -1) throw std::runtime_error("Could not open file: " + filename);
        struct stat sb;
        if (fstat(fd, &sb) == -1) { close(fd); throw std::runtime_error("Could not stat file"); }
        file_size = sb.st_size;
        int flags = MAP_SHARED;
        #ifdef __linux__
        flags |= MAP_POPULATE; 
        #endif
        data_ptr = (uint8_t*)mmap(NULL, file_size, PROT_READ, flags, fd, 0);
        if (data_ptr == MAP_FAILED) { close(fd); throw std::runtime_error("mmap failed"); }
        madvise(data_ptr, file_size, MADV_RANDOM);
        offsets = idx_offsets;
        lengths = idx_lengths;
    }
    ~VVTKReader() {
        if (data_ptr) munmap(data_ptr, file_size);
        if (fd != -1) close(fd);
    }
    inline std::pair<uint8_t*, int64_t> get_raw_ptr(int64_t index) {
        if (index >= offsets.size()) return {nullptr, 0};
        return {data_ptr + offsets[index], lengths[index]};
    }
    void prefetch_block(int64_t start_idx, int64_t count) {
        if (start_idx >= offsets.size()) return;
        int64_t end_idx = std::min((int64_t)offsets.size() - 1, start_idx + count - 1);
        size_t mem_start = offsets[start_idx];
        size_t mem_end = offsets[end_idx] + lengths[end_idx];
        madvise(data_ptr + mem_start, mem_end - mem_start, MADV_WILLNEED);
    }
    inline uint8_t* base_ptr() { return data_ptr; }
    torch::Tensor get_blob_view(int64_t index) {
        auto [ptr, len] = get_raw_ptr(index);
        return torch::from_blob(ptr, {len}, torch::TensorOptions().dtype(torch::kUInt8));
    }
    int64_t len() { return offsets.size(); }
};

// ============================================================================
// PART 2: GENERIC VVTK LOADER
// ============================================================================
class VVTKLoader {
private:
    int batch_size;
    int num_workers;
    int ring_size;
    int num_items;
    
    std::vector<size_t> item_dim_elements;
    std::vector<size_t> item_bytes_per_sample;
    std::vector<size_t> item_element_size;
    std::vector<float>  item_pads;
    std::vector<int64_t> item_dtypes;
    std::vector<int> item_comp_modes;  // per-item compression mode
    
    std::vector<std::pair<int, int64_t>> index_map;
    std::vector<std::shared_ptr<VVTKReader>> shards;
    size_t dataset_size;
    bool shuffle_;
    std::mt19937 rng_;

    std::atomic<size_t> global_cursor;
    bool shutdown;

    std::vector<torch::Tensor> buffers_data;
    std::vector<torch::Tensor> buffers_lengths;
    
    std::unique_ptr<std::atomic<int>[]> batch_status; 
    std::mutex queue_mutex;
    std::condition_variable cv_producer;
    std::condition_variable cv_consumer;
    std::vector<std::thread> workers;
    size_t head_ptr;
    int held_slot;

public:
    VVTKLoader(
        std::vector<std::shared_ptr<VVTKReader>> readers,
        std::vector<std::pair<int, int64_t>> global_index_map,
        std::vector<std::vector<int64_t>> shapes,
        std::vector<int64_t> dtypes,
        std::vector<float> pads,
        std::vector<int> comp_modes,
        int batch_size, int num_workers, int ring_size,
        bool shuffle = false
    ) : shards(readers), index_map(global_index_map),
        batch_size(batch_size), num_workers(num_workers), ring_size(ring_size),
        dataset_size(global_index_map.size()), shuffle_(shuffle),
        rng_(std::random_device{}()),
        global_cursor(0), shutdown(false), head_ptr(0), held_slot(-1)
    {
        #ifdef _OPENMP
        omp_set_num_threads(1);
        #endif

        num_items = shapes.size();
        item_dtypes = dtypes;
        item_comp_modes = comp_modes;

        if (dtypes.size() != num_items || pads.size() != num_items ||
            comp_modes.size() != num_items) {
            throw std::runtime_error("Mismatch in shapes/dtypes/pads/comp_modes sizes");
        }

        for (int i = 0; i < num_items; ++i) {
            size_t elements = 1;
            for (auto s : shapes[i]) elements *= s;
            item_dim_elements.push_back(elements);

            size_t dtype_size = get_dtype_size(dtypes[i]);
            item_element_size.push_back(dtype_size);
            item_bytes_per_sample.push_back(elements * dtype_size);
            item_pads.push_back(pads[i]);

            std::vector<int64_t> full_shape = {(int64_t)ring_size, (int64_t)batch_size};
            full_shape.insert(full_shape.end(), shapes[i].begin(), shapes[i].end());
            
            auto opts = torch::TensorOptions()
                .dtype(get_torch_dtype(dtypes[i]))
                .pinned_memory(true);
            
            buffers_data.push_back(torch::empty(full_shape, opts));

            // FIX: Use zeros() instead of empty() for length buffer to avoid garbage
            // if padding loop logic has edge cases or race conditions.
            buffers_lengths.push_back(torch::zeros({ring_size, batch_size}, 
                torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true)));
        }

        batch_status = std::make_unique<std::atomic<int>[]>(ring_size);
        for(int i=0; i<ring_size; ++i) batch_status[i].store(0);

        for(int i=0; i<num_workers; ++i) workers.emplace_back(&VVTKLoader::worker_loop, this);
    }

    ~VVTKLoader() {
        { std::unique_lock<std::mutex> lock(queue_mutex); shutdown = true; }
        cv_producer.notify_all(); cv_consumer.notify_all();
        for(auto& t : workers) if(t.joinable()) t.join();
    }

    std::vector<std::pair<torch::Tensor, torch::Tensor>> next() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        if (held_slot != -1) {
            batch_status[held_slot].store(0);
            cv_producer.notify_all(); 
            held_slot = -1;
        }

        cv_consumer.wait(lock, [this] { return batch_status[head_ptr].load() == 2 || shutdown; });

        if (shutdown && batch_status[head_ptr].load() != 2) return {};

        std::vector<std::pair<torch::Tensor, torch::Tensor>> result;
        result.reserve(num_items);
        
        for(int i=0; i<num_items; ++i) {
            result.push_back({
                buffers_data[i][head_ptr], 
                buffers_lengths[i][head_ptr]
            });
        }

        held_slot = head_ptr;
        head_ptr = (head_ptr + 1) % ring_size;
        
        return result;
    }

    void reset() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (held_slot != -1) {
            batch_status[held_slot].store(0);
            held_slot = -1;
        }

        // Wait for any in-flight workers (status=1) to finish
        cv_consumer.wait(lock, [this] {
            for (int i = 0; i < ring_size; ++i) {
                if (batch_status[i].load() == 1) return false;
            }
            return true;
        });

        // Now safe: no workers are reading index_map
        global_cursor = 0; 

        if (shuffle_) {
            for (size_t i = dataset_size - 1; i > 0; --i) {
                std::uniform_int_distribution<size_t> dist(0, i);
                std::swap(index_map[i], index_map[dist(rng_)]);
            }
        }
        
        for(int i=0; i<ring_size; ++i) {
            batch_status[i].store(0);
        }
        head_ptr = 0;
        cv_producer.notify_all();
    }
    
    int64_t len() { return (dataset_size + batch_size - 1) / batch_size; }

private:
    void pad_buffer(uint8_t* ptr, size_t offset_elements, size_t total_elements, int64_t dtype_code, float pad_val) {
        switch(dtype_code) {
            case 0: { 
                uint8_t v = (uint8_t)pad_val;
                std::fill((uint8_t*)ptr + offset_elements, (uint8_t*)ptr + total_elements, v);
                break;
            }
            case 1: { 
                int8_t v = (int8_t)pad_val;
                std::fill((int8_t*)ptr + offset_elements, (int8_t*)ptr + total_elements, v);
                break;
            }
            case 2: { 
                int16_t v = (int16_t)pad_val;
                std::fill((int16_t*)ptr + offset_elements, (int16_t*)ptr + total_elements, v);
                break;
            }
            case 3: { 
                int32_t v = (int32_t)pad_val;
                std::fill((int32_t*)ptr + offset_elements, (int32_t*)ptr + total_elements, v);
                break;
            }
            case 4: { 
                int64_t v = (int64_t)pad_val;
                std::fill((int64_t*)ptr + offset_elements, (int64_t*)ptr + total_elements, v);
                break;
            }
            case 10: { 
                c10::Half v = (c10::Half)pad_val;
                std::fill((c10::Half*)ptr + offset_elements, (c10::Half*)ptr + total_elements, v);
                break;
            }
            case 11: { 
                float v = (float)pad_val;
                std::fill((float*)ptr + offset_elements, (float*)ptr + total_elements, v);
                break;
            }
            case 12: { 
                double v = (double)pad_val;
                std::fill((double*)ptr + offset_elements, (double*)ptr + total_elements, v);
                break;
            }
            default: break;
        }
    }

    void worker_loop() {
        while (true) {
            size_t my_batch_idx = 0;
            size_t start_idx = 0;

            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv_producer.wait(lock, [this] {
                    if (shutdown) return true;
                    size_t target_slot = (global_cursor / batch_size) % ring_size;
                    return batch_status[target_slot].load() == 0;
                });
                if (shutdown) break;
                
                if (global_cursor >= dataset_size) continue;

                start_idx = global_cursor;
                my_batch_idx = (start_idx / batch_size) % ring_size;
                
                if (batch_status[my_batch_idx].load() != 0) continue;

                global_cursor += batch_size;
                batch_status[my_batch_idx].store(1);
            }

            size_t actual_bs = std::min((size_t)batch_size, dataset_size - start_idx);
            
            // Prefetch (only useful when not shuffled — consecutive samples in same shard)
            if (!shuffle_ && actual_bs > 0) {
                int first_shard = index_map[start_idx].first;
                int last_shard = index_map[start_idx + actual_bs - 1].first;
                if (first_shard == last_shard) {
                    shards[first_shard]->prefetch_block(index_map[start_idx].second, actual_bs);
                }
            }

            for (size_t i = 0; i < actual_bs; ++i) {
                size_t curr_global = start_idx + i;
                auto& idx_info = index_map[curr_global];
                auto [blob_ptr, blob_len] = shards[idx_info.first]->get_raw_ptr(idx_info.second);
                
                if (!blob_ptr) continue;

                uint8_t* curr_ptr = blob_ptr;
                size_t remaining_len = blob_len;

                for (int item = 0; item < num_items; ++item) {
                    if (remaining_len < 64) break; 

                    int64_t* header = (int64_t*)curr_ptr;
                    int64_t code = header[0];
                    int64_t ndim = header[1];
                    
                    size_t numel = 1;
                    for(int k=0; k<ndim; ++k) numel *= header[2+k];
                    size_t item_bytes = numel * get_dtype_size(code);
                    
                    uint8_t* dst_base = (uint8_t*)buffers_data[item][my_batch_idx].data_ptr();
                    size_t esize = item_element_size[item];
                    uint8_t* dst_sample = dst_base + (i * item_dim_elements[item] * esize);
                    int64_t* len_ptr = buffers_lengths[item][my_batch_idx].data_ptr<int64_t>();

                    size_t total_block_size = 0;

                    if (item_comp_modes[item] == COMP_ZSTD) {
                        // Compressed layout: [64-byte header][8-byte compressed_size][compressed data][align]
                        int64_t compressed_size = *(int64_t*)(curr_ptr + 64);
                        uint8_t* compressed_data = curr_ptr + 64 + 8;

                        if (item_bytes <= item_bytes_per_sample[item]) {
                            // Common case: decompress directly into ring buffer
                            size_t res = decompress_zstd_to(compressed_data, compressed_size,
                                                            dst_sample, item_bytes_per_sample[item]);
                            if (res == 0) {
                                std::memset(dst_sample, 0, item_bytes_per_sample[item]);
                            }
                        } else {
                            // Decompressed data larger than target buffer: use temp
                            thread_local std::vector<uint8_t> decomp_buf;
                            decomp_buf.resize(item_bytes);
                            size_t res = decompress_zstd_to(compressed_data, compressed_size,
                                                            decomp_buf.data(), item_bytes);
                            if (res > 0) {
                                std::memcpy(dst_sample, decomp_buf.data(), item_bytes_per_sample[item]);
                            } else {
                                std::memset(dst_sample, 0, item_bytes_per_sample[item]);
                            }
                        }

                        // Pad remaining elements if decompressed data is smaller than buffer
                        if (item_bytes < item_bytes_per_sample[item]) {
                            size_t elems_copied = item_bytes / esize;
                            pad_buffer(dst_sample, elems_copied, item_dim_elements[item],
                                       item_dtypes[item], item_pads[item]);
                        }

                        total_block_size = 64 + 8 + compressed_size;
                    } else if (item_comp_modes[item] == COMP_FLAC) {
                        // FLAC layout: [64-byte header][8-byte compressed_size][FLAC bitstream][align]
                        int64_t compressed_size = *(int64_t*)(curr_ptr + 64);
                        uint8_t* compressed_data = curr_ptr + 64 + 8;
                        size_t max_frames = item_dim_elements[item];
                        size_t frames_read = 0;

                        if (item_dtypes[item] == 11) {
                            frames_read = decompress_flac_f32(compressed_data, compressed_size,
                                                              (float*)dst_sample, max_frames);
                        } else if (item_dtypes[item] == 2) {
                            frames_read = decompress_flac_s16(compressed_data, compressed_size,
                                                              (int16_t*)dst_sample, max_frames);
                        } else if (item_dtypes[item] == 3) {
                            frames_read = decompress_flac_s32(compressed_data, compressed_size,
                                                              (int32_t*)dst_sample, max_frames);
                        } else {
                            // Unsupported dtype for FLAC — decode to float32 into temp, then cast
                            thread_local std::vector<float> flac_tmp;
                            flac_tmp.resize(max_frames);
                            frames_read = decompress_flac_f32(compressed_data, compressed_size,
                                                              flac_tmp.data(), max_frames);
                            size_t copy_bytes = std::min(frames_read * sizeof(float), item_bytes_per_sample[item]);
                            std::memcpy(dst_sample, flac_tmp.data(), copy_bytes);
                        }

                        if (frames_read == 0) {
                            std::memset(dst_sample, 0, item_bytes_per_sample[item]);
                        }

                        numel = frames_read;
                        item_bytes = frames_read * esize;

                        // Pad remaining elements
                        if (item_bytes < item_bytes_per_sample[item]) {
                            size_t elems_copied = item_bytes / esize;
                            pad_buffer(dst_sample, elems_copied, item_dim_elements[item],
                                       item_dtypes[item], item_pads[item]);
                        }

                        total_block_size = 64 + 8 + compressed_size;
                    } else {
                        // Uncompressed layout: [64-byte header][raw payload][align]
                        uint8_t* payload = curr_ptr + 64;
                        size_t bytes_to_copy = std::min(item_bytes, item_bytes_per_sample[item]);
                        std::memcpy(dst_sample, payload, bytes_to_copy);

                        if (bytes_to_copy < item_bytes_per_sample[item]) {
                            size_t elems_copied = bytes_to_copy / esize;
                            pad_buffer(dst_sample, elems_copied, item_dim_elements[item],
                                       item_dtypes[item], item_pads[item]);
                        }

                        total_block_size = 64 + item_bytes;
                    }

                    len_ptr[i] = numel;

                    if (total_block_size % 8 != 0) total_block_size += (8 - (total_block_size % 8));
                    curr_ptr += total_block_size;
                    if (total_block_size > remaining_len) remaining_len = 0;
                    else remaining_len -= total_block_size;
                }
            }
            
            // Pad remainder of the batch
             if (actual_bs < (size_t)batch_size) {
                 for(int i = (int)actual_bs; i < batch_size; ++i) {
                     for(int item=0; item < num_items; ++item) {
                         uint8_t* dst_base = (uint8_t*)buffers_data[item][my_batch_idx].data_ptr();
                         size_t esize = item_element_size[item];
                         uint8_t* dst_sample = dst_base + (i * item_dim_elements[item] * esize);
                         int64_t* len_ptr = buffers_lengths[item][my_batch_idx].data_ptr<int64_t>();
                         
                         pad_buffer(dst_sample, 0, item_dim_elements[item], item_dtypes[item], item_pads[item]);
                         len_ptr[i] = 0;
                     }
                 }
             }

            { 
                std::unique_lock<std::mutex> lock(queue_mutex); 
                batch_status[my_batch_idx].store(2);
            }
            cv_consumer.notify_one();
        }
    }
};

// ============================================================================
// PART 3: DECODE BLOB ITEMS — for VVTKDataset.__getitem__ (Python side)
// Parses a blob and returns a vector of tensors, one per item.
// For uncompressed data, returns a zero-copy view when possible.
// For compressed data, decompresses into a new tensor.
// ============================================================================
std::vector<torch::Tensor> decode_blob_items(
    torch::Tensor blob,
    std::vector<int> comp_modes
) {
    int num_items = comp_modes.size();
    uint8_t* blob_ptr = blob.data_ptr<uint8_t>();
    int64_t blob_len = blob.numel();
    size_t offset = 0;

    std::vector<torch::Tensor> results;
    results.reserve(num_items);

    for (int item = 0; item < num_items; ++item) {
        if ((int64_t)(offset + 64) > blob_len) {
            throw std::runtime_error("Corrupt blob: not enough bytes for header");
        }

        // 1. Parse header
        int64_t* header = (int64_t*)(blob_ptr + offset);
        int64_t code = header[0];
        int64_t ndim = header[1];
        std::vector<int64_t> shape(ndim);
        size_t numel = 1;
        for (int k = 0; k < ndim; ++k) {
            shape[k] = header[2 + k];
            numel *= shape[k];
        }
        size_t esize = get_dtype_size(code);
        size_t item_bytes = numel * esize;
        auto torch_dtype = get_torch_dtype(code);

        size_t total_block_size = 0;
        torch::Tensor tensor;

        if (comp_modes[item] == COMP_NONE) {
            // Zero-copy view into the mmap'd blob
            uint8_t* payload = blob_ptr + offset + 64;
            tensor = torch::from_blob(payload, shape,
                         torch::TensorOptions().dtype(torch_dtype));
            total_block_size = 64 + item_bytes;

        } else if (comp_modes[item] == COMP_ZSTD) {
            int64_t compressed_size = *(int64_t*)(blob_ptr + offset + 64);
            uint8_t* compressed_data = blob_ptr + offset + 64 + 8;

            tensor = torch::empty(shape, torch::TensorOptions().dtype(torch_dtype));
            size_t res = decompress_zstd_to(compressed_data, compressed_size,
                                             (uint8_t*)tensor.data_ptr(), item_bytes);
            if (res == 0) {
                throw std::runtime_error("ZSTD decompression failed");
            }
            total_block_size = 64 + 8 + compressed_size;

        } else if (comp_modes[item] == COMP_FLAC) {
            int64_t compressed_size = *(int64_t*)(blob_ptr + offset + 64);
            uint8_t* compressed_data = blob_ptr + offset + 64 + 8;

            // FLAC always decodes to float32
            // numel from header = actual sample count
            tensor = torch::empty(shape, torch::TensorOptions().dtype(torch::kFloat32));
            size_t frames_read = decompress_flac_f32(
                compressed_data, compressed_size,
                tensor.data_ptr<float>(), numel);
            if (frames_read == 0) {
                throw std::runtime_error("FLAC decompression failed");
            }
            // Trim if fewer frames than expected
            if ((int64_t)frames_read < (int64_t)numel) {
                tensor = tensor.narrow(0, 0, frames_read);
            }
            total_block_size = 64 + 8 + compressed_size;

        } else {
            throw std::runtime_error("Unknown compression mode: " + std::to_string(comp_modes[item]));
        }

        results.push_back(tensor);

        // Advance with alignment
        if (total_block_size % 8 != 0)
            total_block_size += (8 - (total_block_size % 8));
        offset += total_block_size;
    }

    return results;
}

PYBIND11_MODULE(_vvtk_core, m) {
    pybind11::class_<VVTKReader, std::shared_ptr<VVTKReader>>(m, "VVTKReader")
        .def(pybind11::init<std::string, std::vector<int64_t>, std::vector<int64_t>>())
        .def("get_blob_view", &VVTKReader::get_blob_view)
        .def("len", &VVTKReader::len);

    pybind11::class_<VVTKLoader>(m, "VVTKLoader")
        .def(pybind11::init<
            std::vector<std::shared_ptr<VVTKReader>>,
            std::vector<std::pair<int, int64_t>>,
            std::vector<std::vector<int64_t>>, 
            std::vector<int64_t>, 
            std::vector<float>, 
            std::vector<int>,
            int, int, int,
            bool
        >())
        .def("next", &VVTKLoader::next)
        .def("reset", &VVTKLoader::reset)
        .def("len", &VVTKLoader::len);

    m.def("decode_blob_items", &decode_blob_items,
          "Decode a blob into a list of tensors, one per item.\n"
          "Supports none/zstd/flac compression.\n"
          "Uncompressed items are returned as zero-copy views.",
          pybind11::arg("blob"), pybind11::arg("comp_modes"));
}