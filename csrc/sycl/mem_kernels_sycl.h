// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include "../kv_transfer_types.h"
#include "../kv_transfer_plan_types.h"

/**
 * Enqueue block gather/scatter on the current XPU stream for device.
 *
 * paged_buffer_ptrs_tensor and block_ids are contiguous XPU int64/uint64 and
 * int64 vectors respectively. Object pointers address contiguous device or
 * USM-host [kv_size, nl, chunk_tokens, nh * hs] buffers. Pointers and buffers
 * must remain alive until the stream completes. Block IDs must be in [0, nb)
 * and unique for H2D. Prefix positions and an incomplete final object's tail
 * are left untouched. No allocation, host copy, or synchronization is done.
 *
 * Pointer order is one entry for cross-layer, K layers then V layers for
 * kv-list, interleaved K/V for per-layer tuples, otherwise one per layer.
 * Variable-width plane tuples require the tensor-form torch fallback.
 *
 * Throws c10::Error for invalid host-visible geometry, metadata, or pointers.
 */
void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    const std::vector<uintptr_t>& lmcache_objects_ptrs,
    const torch::Tensor& block_ids, const torch::Device& device,
    TransferDirection direction, PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size, EngineKVFormat engine_kv_format,
    int skip_prefix_n_blocks);

/**
 * Enqueue token-slot gather/scatter on the current XPU stream.
 *
 * Packed NHD/HND engine rows contain each head's K followed by V. key_value
 * retains the V2 contiguous [2, nl, tokens, nh * hs] representation, where
 * head_size is the component hs (half the packed engine content width).
 * key_value_ptrs and slot_mapping are contiguous XPU uint64/int64 and int64
 * vectors. Pointers address normalized layer views, including storage offsets.
 * An unindexed paged_memory_device uses the pointer tensor's XPU device.
 * block_stride_elems is the physical packed-engine block stride; zero selects
 * 2 * block_size * nh * hs. Interior block dimensions must be tightly packed.
 *
 * Negative slots and skip_prefix_n_tokens leave destinations untouched.
 * key_value must be XPU or device-accessible pinned host memory; all operands
 * must stay alive until the stream completes. Copies do not cross component
 * boundaries and use byte copies for misaligned engine addresses.
 *
 * Legacy and MLA formats retain their existing addressing and ignore
 * head_size; they require block_stride_elems == 0.
 * Throws c10::Error for invalid packed geometry or tensor metadata.
 */
void multi_layer_kv_transfer(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const EngineKVFormat engine_kv_format, const int block_size = 0,
    const int head_size = 0, const int skip_prefix_n_tokens = 0,
    const int64_t block_stride_elems = 0);

void single_layer_kv_transfer(torch::Tensor& lmc_key_value_cache,
                              torch::Tensor& vllm_key_value_cache,
                              torch::Tensor& slot_mapping,
                              const TransferDirection direction,
                              const EngineKVFormat engine_kv_format,
                              const bool token_major = false);

// Asynchronous memory copy between host and device buffers.
// The `direction` parameter is retained for API compatibility but is unused:
// SYCL USM memcpy infers direction from pointer allocation types.
void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments);

// deprecated / unused except in unit tests
void load_and_reshape_flash(torch::Tensor& key_value, torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping, const int layer_idx);

// deprecated / unused except in unit tests
void reshape_and_cache_back_flash(torch::Tensor& key_value,
                                  torch::Tensor& key_cache,
                                  torch::Tensor& value_cache,
                                  torch::Tensor& slot_mapping,
                                  const int layer_idx);

// SYCL/XPU pinned (USM host) allocation. Same name/signature as the CUDA
// alloc_pinned_ptr (csrc/cuda/mem_alloc.cpp) so lmcache._get_backend()
// overrides it by name; `flags` is accepted for signature parity and ignored on
// XPU.
uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags = 0);
void free_pinned_ptr(uintptr_t ptr);
