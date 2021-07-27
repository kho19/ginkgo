/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_COMMON_BASE_KERNEL_LAUNCH_REDUCTION_HPP_
#error \
    "This file can only be used from inside common/base/kernel_launch_reduction.hpp"
#endif


#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/device_guard.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
__global__ __launch_bounds__(
    default_block_size) void generic_kernel_reduction_1d(int64 size,
                                                         KernelFunction fn,
                                                         ReductionOp op,
                                                         FinalizeOp finalize,
                                                         ValueType init,
                                                         ValueType *storage,
                                                         KernelArgs... args)
{
    __shared__
        UninitializedArray<ValueType, default_block_size / config::warp_size>
            warp_partial;
    static_assert(default_block_size / config::warp_size <= config::warp_size,
                  "needs third reduction level");
    auto tidx = thread::get_thread_id_flat<int64>();
    auto grid_size = thread::get_thread_num_flat<int64>();
    auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    auto partial = init;
    for (int64 i = tidx; i < size; i += grid_size) {
        partial = op(partial, fn(i, args...));
    }
    partial = reduce(warp, partial, op);
    if (warp.thread_rank() == 0) {
        warp_partial[threadIdx.x / config::warp_size] = partial;
    }
    __syncthreads();
    if (threadIdx.x < config::warp_size) {
        partial = reduce(warp,
                         threadIdx.x < default_block_size / config::warp_size
                             ? warp_partial[threadIdx.x]
                             : init,
                         op);
        if (threadIdx.x == 0) {
            storage[blockIdx.x] = finalize(partial);
        }
    }
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
__global__ __launch_bounds__(
    default_block_size) void generic_kernel_reduction_2d(int64 rows, int64 cols,
                                                         KernelFunction fn,
                                                         ReductionOp op,
                                                         FinalizeOp finalize,
                                                         ValueType init,
                                                         ValueType *storage,
                                                         KernelArgs... args)
{
    __shared__
        UninitializedArray<ValueType, default_block_size / config::warp_size>
            warp_partial;
    static_assert(default_block_size / config::warp_size <= config::warp_size,
                  "needs third reduction level");
    auto tidx = thread::get_thread_id_flat<int64>();
    auto grid_size = thread::get_thread_num_flat<int64>();
    auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    auto partial = init;
    for (int64 i = tidx; i < rows * cols; i += grid_size) {
        const auto row = i / cols;
        const auto col = i % cols;
        partial = op(partial, fn(row, col, args...));
    }
    partial = reduce(warp, partial, op);
    if (warp.thread_rank() == 0) {
        warp_partial[threadIdx.x / config::warp_size] = partial;
    }
    __syncthreads();
    if (threadIdx.x < config::warp_size) {
        partial = reduce(warp,
                         threadIdx.x < default_block_size / config::warp_size
                             ? warp_partial[threadIdx.x]
                             : init,
                         op);
        if (threadIdx.x == 0) {
            storage[blockIdx.x] = finalize(partial);
        }
    }
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction(std::shared_ptr<const HipExecutor> exec,
                          KernelFunction fn, ReductionOp op,
                          FinalizeOp finalize, ValueType init,
                          ValueType *result, size_type size,
                          KernelArgs &&...args)
{
    constexpr int oversubscription = 16;
    gko::hip::device_guard guard{exec->get_device_id()};
    constexpr auto block_size = default_block_size;
    const auto num_blocks = std::min<int64>(
        ceildiv(size, block_size), exec->get_num_warps() * oversubscription);
    if (num_blocks > 1) {
        Array<ValueType> partial{exec, static_cast<size_type>(num_blocks)};
        hipLaunchKernelGGL(
            generic_kernel_reduction_1d, num_blocks, block_size, 0, 0,
            static_cast<int64>(size), fn, op,
            [] __device__(auto v) { return v; }, as_hip_type(init),
            as_hip_type(partial.get_data()), map_to_device(args)...);
        hipLaunchKernelGGL(
            generic_kernel_reduction_1d, 1, block_size, 0, 0,
            static_cast<int64>(num_blocks),
            [] __device__(auto i, auto v) { return v[i]; }, op, finalize,
            as_hip_type(init), as_hip_type(result),
            as_hip_type(partial.get_const_data()));
    } else {
        hipLaunchKernelGGL(generic_kernel_reduction_1d, 1, block_size, 0, 0,
                           static_cast<int64>(size), fn, op, finalize,
                           as_hip_type(init), as_hip_type(result),
                           map_to_device(args)...);
    }
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction(std::shared_ptr<const HipExecutor> exec,
                          KernelFunction fn, ReductionOp op,
                          FinalizeOp finalize, ValueType init,
                          ValueType *result, dim<2> size, KernelArgs &&...args)
{
    constexpr int oversubscription = 16;
    gko::hip::device_guard guard{exec->get_device_id()};
    constexpr auto block_size = default_block_size;
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto num_blocks =
        std::min<int64>(ceildiv(rows * cols, block_size),
                        exec->get_num_warps() * oversubscription);
    if (num_blocks > 1) {
        Array<ValueType> partial{exec, static_cast<size_type>(num_blocks)};
        hipLaunchKernelGGL(
            generic_kernel_reduction_2d, num_blocks, block_size, 0, 0, rows,
            cols, fn, op, [] __device__(auto v) { return v; },
            as_hip_type(init), as_hip_type(partial.get_data()),
            map_to_device(args)...);
        hipLaunchKernelGGL(
            generic_kernel_reduction_1d, 1, block_size, 0, 0,
            static_cast<int64>(num_blocks),
            [] __device__(auto i, auto v) { return v[i]; }, op, finalize,
            as_hip_type(init), as_hip_type(result),
            as_hip_type(partial.get_const_data()));
    } else {
        hipLaunchKernelGGL(generic_kernel_reduction_2d, 1, block_size, 0, 0,
                           rows, cols, fn, op, finalize, as_hip_type(init),
                           as_hip_type(result), map_to_device(args)...);
    }
}


template <int subwarp_size, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp, typename... KernelArgs>
__global__
    __launch_bounds__(default_block_size) void generic_kernel_row_reduction_2d(
        int64 rows, int64 cols, int64 col_blocks, KernelFunction fn,
        ReductionOp op, FinalizeOp finalize, ValueType init, ValueType *result,
        int64 result_stride, KernelArgs... args)
{
    const auto idx = thread::get_subwarp_id_flat<subwarp_size, int64>();
    const auto row = idx % rows;
    const auto col_block = idx / rows;
    if (col_block >= col_blocks) {
        return;
    }
    const auto cols_per_part =
        ceildiv(ceildiv(cols, subwarp_size), col_blocks) * subwarp_size;
    const auto begin = cols_per_part * col_block;
    const auto end = min(begin + cols_per_part, cols);
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto partial = init;
    for (auto col = begin + subwarp.thread_rank(); col < end;
         col += subwarp_size) {
        partial = op(partial, fn(row, col, args...));
    }
    partial = reduce(subwarp, partial, op);
    result[(row + col_block * rows) * result_stride] = finalize(partial);
}


template <int subwarp_size, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp, typename... KernelArgs>
__global__
    __launch_bounds__(default_block_size) void generic_kernel_col_reduction_2d_small(
        int64 rows, int64 cols, KernelFunction fn, ReductionOp op,
        FinalizeOp finalize, ValueType init, ValueType *result,
        KernelArgs... args)
{
    constexpr auto warp_size = config::warp_size;
    constexpr auto warps_per_block = default_block_size / warp_size;
    // stores the subwarp_size partial sums from each warp, grouped by warp
    constexpr auto shared_storage = warps_per_block * subwarp_size;
    __shared__ UninitializedArray<ValueType, shared_storage> block_partial;
    const auto subwarp_id = thread::get_subwarp_id_flat<subwarp_size, int64>();
    const auto local_warp_id = threadIdx.x / warp_size;
    const auto local_subwarp_id = threadIdx.x % warp_size / subwarp_size;
    const auto subwarp_num =
        thread::get_subwarp_num_flat<subwarp_size, int64>();
    const auto block = group::this_thread_block();
    //
    if (threadIdx.x < shared_storage) {
        block_partial[threadIdx.x] = init;
    }
    block.sync();
    //
    const auto warp = group::tiled_partition<warp_size>(block);
    const auto warp_rank = warp.thread_rank();
    const auto subwarp_rank = warp_rank % subwarp_size;
    const auto col = static_cast<int64>(subwarp_rank);
    auto partial = init;
    // accumulate within a thread
    if (col < cols) {
        for (auto row = subwarp_id; row < rows; row += subwarp_num) {
            partial = op(partial, fn(row, col, args...));
        }
    }
    // accumulate between all subwarps in the warp
#pragma unroll
    for (unsigned i = subwarp_size; i < warp_size; i *= 2) {
        partial = op(partial, warp.shfl_xor(partial, i));
    }  // store the result to shared memory
    if (local_subwarp_id == 0) {
        block_partial[local_warp_id * subwarp_size + subwarp_rank] = partial;
    }
    block.sync();
    // in a single thread: accumulate the results
    if (local_warp_id == 0) {
        partial = init;
        // accumulate the partial results within a thread
        if (shared_storage >= warp_size) {
#pragma unroll
            for (int i = 0; i < shared_storage; i += warp_size) {
                partial = op(partial, block_partial[i + warp_rank]);
            }
        } else if (warp_rank < shared_storage) {
            partial = op(partial, block_partial[warp_rank]);
        }
        // accumulate between all subwarps in the warp
#pragma unroll
        for (unsigned i = subwarp_size; i < warp_size; i *= 2) {
            partial = op(partial, warp.shfl_xor(partial, i));
        }
        if (warp_rank < cols) {
            result[warp_rank + blockIdx.x * cols] = finalize(partial);
        }
    }
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
__global__
    __launch_bounds__(default_block_size) void generic_kernel_col_reduction_2d_blocked(
        int64 rows, int64 cols, KernelFunction fn, ReductionOp op,
        FinalizeOp finalize, ValueType init, ValueType *result,
        KernelArgs... args)
{
    constexpr auto warp_size = config::warp_size;
    __shared__ UninitializedArray<ValueType, default_block_size> block_partial;
    const auto warp_id = thread::get_subwarp_id_flat<warp_size, int64>();
    const auto warp_num = thread::get_subwarp_num_flat<warp_size, int64>();
    const auto block = group::this_thread_block();
    const auto warp = group::tiled_partition<warp_size>(block);
    const auto warp_rank = warp.thread_rank();
    const auto col = warp_rank + static_cast<int64>(blockIdx.y) * warp_size;
    auto partial = init;
    // accumulate within a thread
    if (col < cols) {
        for (auto row = warp_id; row < rows; row += warp_num) {
            partial = op(partial, fn(row, col, args...));
        }
    }
    block_partial[threadIdx.x] = partial;
    block.sync();
    // in a single warp: accumulate the results
    if (threadIdx.x < warp_size) {
        partial = init;
        // accumulate the partial results within a thread
#pragma unroll
        for (int i = 0; i < default_block_size; i += warp_size) {
            partial = op(partial, block_partial[i + warp_rank]);
        }
        if (col < cols) {
            result[col + blockIdx.x * cols] = finalize(partial);
        }
    }
}


template <typename ValueType, typename ReductionOp, typename FinalizeOp>
__global__
    __launch_bounds__(default_block_size) void generic_kernel_reduction_finalize_2d(
        int64 num_results, int64 num_blocks, ReductionOp op,
        FinalizeOp finalize, ValueType init, const ValueType *input,
        int64 result_stride, ValueType *result)
{
    const auto idx = thread::get_thread_id_flat<int64>();
    if (idx >= num_results) {
        return;
    }
    auto partial = init;
    for (int64 block = 0; block < num_blocks; block++) {
        partial = op(partial, input[idx + block * num_results]);
    }
    result[idx * result_stride] = finalize(partial);
}


namespace {


template <int subwarp_size, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp, typename... KernelArgs>
void run_generic_kernel_row_reduction(syn::value_list<int, subwarp_size>,
                                      int64 rows, int64 cols, int64 col_blocks,
                                      KernelFunction fn, ReductionOp op,
                                      FinalizeOp finalize, ValueType init,
                                      ValueType *result, int64 result_stride,
                                      KernelArgs... args)
{
    const auto num_blocks =
        ceildiv(rows * col_blocks * subwarp_size, default_block_size);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(generic_kernel_row_reduction_2d<subwarp_size>),
        num_blocks, default_block_size, 0, 0, rows, cols, col_blocks, fn, op,
        finalize, as_hip_type(init), as_hip_type(result), result_stride,
        args...);
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_run_generic_kernel_row_reduction,
                                    run_generic_kernel_row_reduction)


template <int subwarp_size, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void run_generic_col_reduction_small(syn::value_list<int, subwarp_size>,
                                     int64 max_blocks,
                                     std::shared_ptr<const HipExecutor> exec,
                                     KernelFunction fn, ReductionOp op,
                                     FinalizeOp finalize, ValueType init,
                                     ValueType *result, dim<2> size,
                                     MappedKernelArgs... args)
{
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto num_blocks = std::min<int64>(
        ceildiv(rows * subwarp_size, default_block_size), max_blocks);
    if (num_blocks <= 1) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                generic_kernel_col_reduction_2d_small<subwarp_size>),
            1, default_block_size, 0, 0, rows, cols, fn, op, finalize,
            as_hip_type(init), as_hip_type(result), args...);
    } else {
        Array<ValueType> tmp_storage{exec,
                                     static_cast<size_type>(num_blocks * cols)};
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                generic_kernel_col_reduction_2d_small<subwarp_size>),
            num_blocks, default_block_size, 0, 0, rows, cols, fn, op,
            [] __device__(auto v) { return v; }, as_hip_type(init),
            as_hip_type(tmp_storage.get_data()), args...);
        hipLaunchKernelGGL(
            generic_kernel_reduction_finalize_2d,
            ceildiv(cols, default_block_size), default_block_size, 0, 0, cols,
            num_blocks, op, finalize, as_hip_type(init),
            as_hip_type(tmp_storage.get_const_data()), 1, as_hip_type(result));
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_generic_col_reduction_small,
                                    run_generic_col_reduction_small);


}  // namespace


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_row_reduction(std::shared_ptr<const HipExecutor> exec,
                              KernelFunction fn, ReductionOp op,
                              FinalizeOp finalize, ValueType init,
                              ValueType *result, size_type result_stride,
                              dim<2> size, KernelArgs &&...args)
{
    using subwarp_sizes =
        syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;
    constexpr int oversubscription = 16;
    gko::hip::device_guard guard{exec->get_device_id()};
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto resources =
        exec->get_num_warps() * config::warp_size * oversubscription;
    if (rows * cols > resources && rows < cols) {
        const auto col_blocks = ceildiv(rows * cols, resources);
        Array<ValueType> partial{exec,
                                 static_cast<size_type>(col_blocks * rows)};
        const auto num_blocks =
            ceildiv(rows * col_blocks * config::warp_size, default_block_size);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(generic_kernel_row_reduction_2d<config::warp_size>),
            num_blocks, default_block_size, 0, 0, rows, cols, col_blocks, fn,
            op, [] __device__(auto v) { return v; }, as_hip_type(init),
            as_hip_type(partial.get_data()), 1, map_to_device(args)...);
        const auto num_finalize_blocks = ceildiv(rows, default_block_size);
        hipLaunchKernelGGL(
            generic_kernel_reduction_finalize_2d, num_finalize_blocks,
            default_block_size, 0, 0, rows, col_blocks, op, finalize,
            as_hip_type(init), as_hip_type(partial.get_const_data()),
            static_cast<int64>(result_stride), as_hip_type(result));
    } else {
        select_run_generic_kernel_row_reduction(
            subwarp_sizes(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size >= cols ||
                       compiled_subwarp_size == config::warp_size;
            },
            syn::value_list<int>(), syn::type_list<>(), rows, cols, 1, fn, op,
            finalize, init, result, static_cast<int64>(result_stride),
            map_to_device(args)...);
    }
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_col_reduction(std::shared_ptr<const HipExecutor> exec,
                              KernelFunction fn, ReductionOp op,
                              FinalizeOp finalize, ValueType init,
                              ValueType *result, dim<2> size,
                              KernelArgs &&...args)
{
    using subwarp_sizes =
        syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;
    constexpr int oversubscription = 16;
    gko::hip::device_guard guard{exec->get_device_id()};
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto max_blocks = exec->get_num_warps() * config::warp_size *
                            oversubscription / default_block_size;
    if (cols <= config::warp_size) {
        select_generic_col_reduction_small(
            subwarp_sizes(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size >= cols ||
                       compiled_subwarp_size == config::warp_size;
            },
            syn::value_list<int>(), syn::type_list<>(), max_blocks, exec, fn,
            op, finalize, init, result, size, map_to_device(args)...);
    } else {
        const auto col_blocks = ceildiv(cols, config::warp_size);
        const auto row_blocks =
            ceildiv(std::min<int64>(
                        ceildiv(rows * config::warp_size, default_block_size),
                        max_blocks),
                    col_blocks);
        if (row_blocks <= 1) {
            hipLaunchKernelGGL(generic_kernel_col_reduction_2d_blocked,
                               dim3(1, col_blocks), default_block_size, 0, 0,
                               rows, cols, fn, op, finalize, as_hip_type(init),
                               as_hip_type(result), map_to_device(args)...);
        } else {
            Array<ValueType> tmp_storage{
                exec, static_cast<size_type>(row_blocks * cols)};
            hipLaunchKernelGGL(
                generic_kernel_col_reduction_2d_blocked,
                dim3(row_blocks, col_blocks), default_block_size, 0, 0, rows,
                cols, fn, op, [] __device__(auto v) { return v; },
                as_hip_type(init), as_hip_type(tmp_storage.get_data()),
                map_to_device(args)...);
            hipLaunchKernelGGL(generic_kernel_reduction_finalize_2d,
                               ceildiv(cols, default_block_size),
                               default_block_size, 0, 0, cols, row_blocks, op,
                               finalize, as_hip_type(init),
                               as_hip_type(tmp_storage.get_const_data()), 1,
                               as_hip_type(result));
        }
    }
}


}  // namespace hip
}  // namespace kernels
}  // namespace gko
