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

namespace kernel {
namespace {


template <int num_thread_per_worker, bool atomic, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType,
          typename Closure>
__device__ void spmv_kernel(
    const size_type num_rows, const int num_worker_per_row,
    acc::range<a_accessor> val, const IndexType* __restrict__ col,
    const size_type stride, const size_type num_stored_elements_per_row,
    acc::range<b_accessor> b, OutputValueType* __restrict__ c,
    const size_type c_stride, Closure op)
{
    const auto tidx = thread::get_thread_id_flat();
    const decltype(tidx) column_id = blockIdx.y;
    if (num_thread_per_worker == 1) {
        // Specialize the num_thread_per_worker = 1. It doesn't need the shared
        // memory, __syncthreads, and atomic_add
        if (tidx < num_rows) {
            auto temp = zero<OutputValueType>();
            for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
                const auto ind = tidx + idx * stride;
                const auto col_idx = col[ind];
                if (col_idx < idx) {
                    break;
                } else {
                    temp += val(ind) * b(col_idx, column_id);
                }
            }
            const auto c_ind = tidx * c_stride + column_id;
            c[c_ind] = op(temp, c[c_ind]);
        }
    } else {
        if (tidx < num_worker_per_row * num_rows) {
            const auto idx_in_worker = threadIdx.y;
            const auto x = tidx % num_rows;
            const auto worker_id = tidx / num_rows;
            const auto step_size = num_worker_per_row * num_thread_per_worker;
            __shared__ UninitializedArray<
                OutputValueType, default_block_size / num_thread_per_worker>
                storage;
            if (idx_in_worker == 0) {
                storage[threadIdx.x] = 0;
            }
            __syncthreads();
            auto temp = zero<OutputValueType>();
            for (size_type idx =
                     worker_id * num_thread_per_worker + idx_in_worker;
                 idx < num_stored_elements_per_row; idx += step_size) {
                const auto ind = x + idx * stride;
                const auto col_idx = col[ind];
                if (col_idx < idx) {
                    break;
                } else {
                    temp += val(ind) * b(col_idx, column_id);
                }
            }
            atomic_add(&storage[threadIdx.x], temp);
            __syncthreads();
            if (idx_in_worker == 0) {
                const auto c_ind = x * c_stride + column_id;
                if (atomic) {
                    atomic_add(&(c[c_ind]), op(storage[threadIdx.x], c[c_ind]));
                } else {
                    c[c_ind] = op(storage[threadIdx.x], c[c_ind]);
                }
            }
        }
    }
}


template <int num_thread_per_worker, bool atomic = false, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void spmv(
    const size_type num_rows, const int num_worker_per_row,
    acc::range<a_accessor> val, const IndexType* __restrict__ col,
    const size_type stride, const size_type num_stored_elements_per_row,
    acc::range<b_accessor> b, OutputValueType* __restrict__ c,
    const size_type c_stride)
{
    spmv_kernel<num_thread_per_worker, atomic>(
        num_rows, num_worker_per_row, val, col, stride,
        num_stored_elements_per_row, b, c, c_stride,
        [](const OutputValueType& x, const OutputValueType& y) { return x; });
}


template <int num_thread_per_worker, bool atomic = false, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void spmv(
    const size_type num_rows, const int num_worker_per_row,
    acc::range<a_accessor> alpha, acc::range<a_accessor> val,
    const IndexType* __restrict__ col, const size_type stride,
    const size_type num_stored_elements_per_row, acc::range<b_accessor> b,
    const OutputValueType* __restrict__ beta, OutputValueType* __restrict__ c,
    const size_type c_stride)
{
    const OutputValueType alpha_val = alpha(0);
    const OutputValueType beta_val = beta[0];
    if (atomic) {
        // Because the atomic operation changes the values of c during
        // computation, it can not directly do alpha * a * b + beta * c
        // operation. The beta * c needs to be done before calling this kernel.
        // Then, this kernel only adds alpha * a * b when it uses atomic
        // operation.
        spmv_kernel<num_thread_per_worker, atomic>(
            num_rows, num_worker_per_row, val, col, stride,
            num_stored_elements_per_row, b, c, c_stride,
            [&alpha_val](const OutputValueType& x, const OutputValueType& y) {
                return alpha_val * x;
            });
    } else {
        spmv_kernel<num_thread_per_worker, atomic>(
            num_rows, num_worker_per_row, val, col, stride,
            num_stored_elements_per_row, b, c, c_stride,
            [&alpha_val, &beta_val](const OutputValueType& x,
                                    const OutputValueType& y) {
                return alpha_val * x + beta_val * y;
            });
    }
}


}  // namespace


template <typename ValueType>
__global__ __launch_bounds__(config::max_block_size) void initialize_zero_dense(
    size_type num_rows, size_type num_cols, size_type stride,
    ValueType* __restrict__ result)
{
    const auto tidx_x = threadIdx.x + blockDim.x * blockIdx.x;
    const auto tidx_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (tidx_x < num_cols && tidx_y < num_rows) {
        result[tidx_y * stride + tidx_x] = zero<ValueType>();
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    size_type num_rows, size_type nnz, size_type source_stride,
    const IndexType* __restrict__ col_idxs,
    const ValueType* __restrict__ values, size_type result_stride,
    ValueType* __restrict__ result)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < num_rows) {
        for (size_type col = 0; col < nnz; col++) {
            result[tidx * result_stride +
                   col_idxs[tidx + col * source_stride]] +=
                values[tidx + col * source_stride];
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void count_nnz_per_row(
    size_type num_rows, size_type max_nnz_per_row, size_type stride,
    const ValueType* __restrict__ values, IndexType* __restrict__ result)
{
    constexpr auto warp_size = config::warp_size;
    const auto row_idx = thread::get_subwarp_id_flat<warp_size>();
    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block());

    if (row_idx < num_rows) {
        IndexType part_result{};
        for (auto i = warp_tile.thread_rank(); i < max_nnz_per_row;
             i += warp_size) {
            if (values[stride * i + row_idx] != zero<ValueType>()) {
                part_result += 1;
            }
        }
        result[row_idx] = reduce(
            warp_tile, part_result,
            [](const size_type& a, const size_type& b) { return a + b; });
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_csr(
    size_type num_rows, size_type max_nnz_per_row, size_type stride,
    const ValueType* __restrict__ source_values,
    const IndexType* __restrict__ source_col_idxs,
    IndexType* __restrict__ result_row_ptrs,
    IndexType* __restrict__ result_col_idxs,
    ValueType* __restrict__ result_values)
{
    const auto tidx = thread::get_thread_id_flat();

    if (tidx < num_rows) {
        auto write_to = result_row_ptrs[tidx];
        for (size_type i = 0; i < max_nnz_per_row; i++) {
            const auto source_idx = tidx + stride * i;
            if (source_values[source_idx] != zero<ValueType>()) {
                result_values[write_to] = source_values[source_idx];
                result_col_idxs[write_to] = source_col_idxs[source_idx];
                write_to++;
            }
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void extract_diagonal(
    size_type diag_size, size_type max_nnz_per_row, size_type orig_stride,
    const ValueType* __restrict__ orig_values,
    const IndexType* __restrict__ orig_col_idxs, ValueType* __restrict__ diag)
{
    const auto tidx = thread::get_thread_id_flat();
    const auto row = tidx % diag_size;
    const auto col = tidx / diag_size;
    const auto ell_ind = orig_stride * col + row;

    if (col < max_nnz_per_row) {
        if (orig_col_idxs[ell_ind] == row &&
            orig_values[ell_ind] != zero<ValueType>()) {
            diag[row] = orig_values[ell_ind];
        }
    }
}


}  // namespace kernel
