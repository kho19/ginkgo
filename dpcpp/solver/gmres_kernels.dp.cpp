/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/solver/gmres_kernels.hpp"


#include <algorithm>
// #include <dpcpp/base/cublas_bindings.hpp>


#include <oneapi/mkl.hpp>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/fill_array.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/math.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The GMRES solver namespace.
 *
 * @ingroup gmres
 */
namespace gmres {


constexpr int default_block_size = 256;
// default_dot_dim can not be 64 in hip because 64 * 64 exceeds their max block
// size limit.
constexpr int default_dot_dim = 16;
constexpr int default_dot_size = default_dot_dim * default_dot_dim;


// #include "common/solver/gmres_kernels.hpp.inc"
// Must be called with at least `max(stride_b * num_rows, krylov_dim *
// num_cols)` threads in total.
template <size_type block_size, typename ValueType>
void initialize_1_kernel(
    size_type num_rows, size_type num_cols, size_type krylov_dim,
    const ValueType *__restrict__ b, size_type stride_b,
    ValueType *__restrict__ residual, size_type stride_residual,
    ValueType *__restrict__ givens_sin, size_type stride_sin,
    ValueType *__restrict__ givens_cos, size_type stride_cos,
    stopping_status *__restrict__ stop_status, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);

    const auto row_idx = global_id / stride_b;
    const auto col_idx = global_id % stride_b;

    if (global_id < num_cols) {
        stop_status[global_id].reset();
    }

    if (row_idx < num_rows && col_idx < num_cols) {
        residual[row_idx * stride_residual + col_idx] =
            b[row_idx * stride_b + col_idx];
    }

    if (global_id < krylov_dim * num_cols) {
        const auto row_givens = global_id / num_cols;
        const auto col_givens = global_id % num_cols;

        givens_sin[row_givens * stride_sin + col_givens] = zero<ValueType>();
        givens_cos[row_givens * stride_cos + col_givens] = zero<ValueType>();
    }
}

template <size_type block_size, typename ValueType>
void initialize_1_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                         sycl::queue *stream, size_type num_rows,
                         size_type num_cols, size_type krylov_dim,
                         const ValueType *b, size_type stride_b,
                         ValueType *residual, size_type stride_residual,
                         ValueType *givens_sin, size_type stride_sin,
                         ValueType *givens_cos, size_type stride_cos,
                         stopping_status *stop_status)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             initialize_1_kernel<block_size>(
                                 num_rows, num_cols, krylov_dim, b, stride_b,
                                 residual, stride_residual, givens_sin,
                                 stride_sin, givens_cos, stride_cos,
                                 stop_status, item_ct1);
                         });
    });
}


// Must be called with at least `num_rows * num_rhs` threads in total.
template <size_type block_size, typename ValueType>
void initialize_2_2_kernel(
    size_type num_rows, size_type num_rhs,
    const ValueType *__restrict__ residual, size_type stride_residual,
    const remove_complex<ValueType> *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norm_collection,
    ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    size_type *__restrict__ final_iter_nums, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_idx = global_id / num_rhs;
    const auto col_idx = global_id % num_rhs;

    if (global_id < num_rhs) {
        residual_norm_collection[global_id] = residual_norm[global_id];
        final_iter_nums[global_id] = 0;
    }

    if (row_idx < num_rows && col_idx < num_rhs) {
        auto value = residual[row_idx * stride_residual + col_idx] /
                     residual_norm[col_idx];
        krylov_bases[row_idx * stride_krylov + col_idx] = value;
    }
}

template <size_type block_size, typename ValueType>
void initialize_2_2_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                           sycl::queue *stream, size_type num_rows,
                           size_type num_rhs, const ValueType *residual,
                           size_type stride_residual,
                           const remove_complex<ValueType> *residual_norm,
                           ValueType *residual_norm_collection,
                           ValueType *krylov_bases, size_type stride_krylov,
                           size_type *final_iter_nums)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             initialize_2_2_kernel<block_size>(
                                 num_rows, num_rhs, residual, stride_residual,
                                 residual_norm, residual_norm_collection,
                                 krylov_bases, stride_krylov, final_iter_nums,
                                 item_ct1);
                         });
    });
}


void increase_final_iteration_numbers_kernel(
    size_type *__restrict__ final_iter_nums,
    const stopping_status *__restrict__ stop_status, size_type total_number,
    sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    if (global_id < total_number) {
        final_iter_nums[global_id] += !stop_status[global_id].has_stopped();
    }
}

void increase_final_iteration_numbers_kernel(dim3 grid, dim3 block,
                                             size_t dynamic_shared_memory,
                                             sycl::queue *stream,
                                             size_type *final_iter_nums,
                                             const stopping_status *stop_status,
                                             size_type total_number)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             increase_final_iteration_numbers_kernel(
                                 final_iter_nums, stop_status, total_number,
                                 item_ct1);
                         });
    });
}


template <typename ValueType>
void multidot_kernel(
    size_type k, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ krylov_bases,
    const ValueType *__restrict__ next_krylov_basis, size_type stride_krylov,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status, sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, default_dot_dim *(default_dot_dim + 1)>
        *reduction_helper_array)
{
    const auto tidx = item_ct1.get_local_id(2);
    const auto tidy = item_ct1.get_local_id(1);
    const auto col_idx = item_ct1.get_group(2) * default_dot_dim + tidx;
    const auto num = ceildiv(num_rows, item_ct1.get_group_range(1));
    const auto start_row = item_ct1.get_group(1) * num;
    const auto end_row = ((item_ct1.get_group(1) + 1) * num > num_rows)
                             ? num_rows
                             : (item_ct1.get_group(1) + 1) * num;
    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`

    ValueType *__restrict__ reduction_helper = (*reduction_helper_array);

    ValueType local_res = zero<ValueType>();
    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        for (size_type i = start_row + tidy; i < end_row;
             i += default_dot_dim) {
            const auto krylov_idx = i * stride_krylov + col_idx;
            local_res +=
                conj(krylov_bases[krylov_idx]) * next_krylov_basis[krylov_idx];
        }
    }
    reduction_helper[tidx * (default_dot_dim + 1) + tidy] = local_res;
    item_ct1.barrier();
    local_res = reduction_helper[tidy * (default_dot_dim + 1) + tidx];
    const auto tile_block = group::tiled_partition<default_dot_dim>(
        group::this_thread_block(item_ct1));
    const auto sum = ::gko::kernels::dpcpp::reduce(
        tile_block, local_res,
        [](const ValueType &a, const ValueType &b) { return a + b; });
    const auto new_col_idx = item_ct1.get_group(2) * default_dot_dim + tidy;
    if (tidx == 0 && new_col_idx < num_cols &&
        !stop_status[new_col_idx].has_stopped()) {
        const auto hessenberg_idx = k * stride_hessenberg + new_col_idx;
        atomic_add(hessenberg_iter + hessenberg_idx, sum);
    }
}

template <typename ValueType>
void multidot_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                     sycl::queue *stream, size_type k, size_type num_rows,
                     size_type num_cols, const ValueType *krylov_bases,
                     const ValueType *next_krylov_basis,
                     size_type stride_krylov, ValueType *hessenberg_iter,
                     size_type stride_hessenberg,
                     const stopping_status *stop_status)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<UninitializedArray<ValueType, default_dot_dim *(
                                                         default_dot_dim + 1)>,
                       0, sycl::access::mode::read_write,
                       sycl::access::target::local>
            reduction_helper_array_acc_ct1(cgh);

        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item_ct1) {
                multidot_kernel(
                    k, num_rows, num_cols, krylov_bases, next_krylov_basis,
                    stride_krylov, hessenberg_iter, stride_hessenberg,
                    stop_status, item_ct1,
                    (UninitializedArray<ValueType, default_dot_dim *(
                                                       default_dot_dim + 1)> *)
                        reduction_helper_array_acc_ct1.get_pointer());
            });
    });
}


// Must be called with at least `num_rows * stride_next_krylov` threads in
// total.
template <int block_size, typename ValueType>
void update_next_krylov_kernel(
    size_type k, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ krylov_bases,
    ValueType *__restrict__ next_krylov_basis, size_type stride_krylov,
    const ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_idx = global_id / stride_krylov;
    const auto col_idx = global_id % stride_krylov;

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto next_krylov_idx = row_idx * stride_krylov + col_idx;
        const auto krylov_idx = row_idx * stride_krylov + col_idx;
        const auto hessenberg_idx = k * stride_hessenberg + col_idx;

        next_krylov_basis[next_krylov_idx] -=
            hessenberg_iter[hessenberg_idx] * krylov_bases[krylov_idx];
    }
}

template <int block_size, typename ValueType>
void update_next_krylov_kernel(
    dim3 grid, dim3 block, size_t dynamic_shared_memory, sycl::queue *stream,
    size_type k, size_type num_rows, size_type num_cols,
    const ValueType *krylov_bases, ValueType *next_krylov_basis,
    size_type stride_krylov, const ValueType *hessenberg_iter,
    size_type stride_hessenberg, const stopping_status *stop_status)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             update_next_krylov_kernel<block_size>(
                                 k, num_rows, num_cols, krylov_bases,
                                 next_krylov_basis, stride_krylov,
                                 hessenberg_iter, stride_hessenberg,
                                 stop_status, item_ct1);
                         });
    });
}


// Must be called with at least `num_cols` blocks, each with `block_size`
// threads. `block_size` must be a power of 2.
template <int block_size, typename ValueType>
void update_hessenberg_2_kernel(
    size_type iter, size_type num_rows, size_type num_cols,
    const ValueType *__restrict__ next_krylov_basis,
    size_type stride_next_krylov, ValueType *__restrict__ hessenberg_iter,
    size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status, sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, block_size> *reduction_helper_array)
{
    const auto tidx = item_ct1.get_local_id(2);
    const auto col_idx = item_ct1.get_group(2);

    // Used that way to get around dynamic initialization warning and
    // template error when using `reduction_helper_array` directly in `reduce`

    ValueType *__restrict__ reduction_helper = (*reduction_helper_array);

    if (col_idx < num_cols && !stop_status[col_idx].has_stopped()) {
        ValueType local_res{};
        for (size_type i = tidx; i < num_rows; i += block_size) {
            const auto next_krylov_idx = i * stride_next_krylov + col_idx;
            const auto next_krylov_value = next_krylov_basis[next_krylov_idx];

            local_res += next_krylov_value * next_krylov_value;
        }

        reduction_helper[tidx] = local_res;

        // Perform thread block reduction. Result is in reduction_helper[0]
        reduce(group::this_thread_block(item_ct1), reduction_helper,
               [](const ValueType &a, const ValueType &b) { return a + b; });

        if (tidx == 0) {
            hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx] =
                sycl::sqrt(reduction_helper[0]);
        }
    }
}

template <int block_size, typename ValueType>
void update_hessenberg_2_kernel(
    dim3 grid, dim3 block, size_t dynamic_shared_memory, sycl::queue *stream,
    size_type iter, size_type num_rows, size_type num_cols,
    const ValueType *next_krylov_basis, size_type stride_next_krylov,
    ValueType *hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *stop_status)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<UninitializedArray<ValueType, block_size>, 0,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
            reduction_helper_array_acc_ct1(cgh);

        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item_ct1) {
                update_hessenberg_2_kernel<block_size>(
                    iter, num_rows, num_cols, next_krylov_basis,
                    stride_next_krylov, hessenberg_iter, stride_hessenberg,
                    stop_status, item_ct1,
                    (UninitializedArray<ValueType, block_size> *)
                        reduction_helper_array_acc_ct1.get_pointer());
            });
    });
}


// Must be called with at least `num_rows * stride_krylov` threads in
// total.
template <int block_size, typename ValueType>
void update_krylov_kernel(
    size_type iter, size_type num_rows, size_type num_cols,
    ValueType *__restrict__ krylov_bases, size_type stride_krylov,
    const ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    const stopping_status *__restrict__ stop_status, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_idx = global_id / stride_krylov;
    const auto col_idx = global_id % stride_krylov;
    const auto hessenberg =
        hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx];

    if (row_idx < num_rows && col_idx < num_cols &&
        !stop_status[col_idx].has_stopped()) {
        const auto krylov_idx = row_idx * stride_krylov + col_idx;

        krylov_bases[krylov_idx] /= hessenberg;
    }
}

template <int block_size, typename ValueType>
void update_krylov_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                          sycl::queue *stream, size_type iter,
                          size_type num_rows, size_type num_cols,
                          ValueType *krylov_bases, size_type stride_krylov,
                          const ValueType *hessenberg_iter,
                          size_type stride_hessenberg,
                          const stopping_status *stop_status)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             update_krylov_kernel<block_size>(
                                 iter, num_rows, num_cols, krylov_bases,
                                 stride_krylov, hessenberg_iter,
                                 stride_hessenberg, stop_status, item_ct1);
                         });
    });
}


template <typename ValueType>
void calculate_sin_and_cos_kernel(size_type col_idx, size_type num_cols,
                                  size_type iter, const ValueType &this_hess,
                                  const ValueType &next_hess,
                                  ValueType *givens_sin, size_type stride_sin,
                                  ValueType *givens_cos, size_type stride_cos,
                                  ValueType &register_sin,
                                  ValueType &register_cos)
{
    if (this_hess == zero<ValueType>()) {
        register_cos = zero<ValueType>();
        register_sin = one<ValueType>();
    } else {
        const auto scale = dpcpp::abs(this_hess) + dpcpp::abs(next_hess);
        const auto hypotenuse =
            scale *
            sycl::sqrt(
                dpcpp::abs(this_hess / scale) * dpcpp::abs(this_hess / scale) +
                dpcpp::abs(next_hess / scale) * dpcpp::abs(next_hess / scale));
        register_cos = conj(this_hess) / hypotenuse;
        register_sin = conj(next_hess) / hypotenuse;
    }
    givens_cos[iter * stride_cos + col_idx] = register_cos;
    givens_sin[iter * stride_sin + col_idx] = register_sin;
}


template <typename ValueType>
void calculate_residual_norm_kernel(size_type col_idx, size_type num_cols,
                                    size_type iter,
                                    const ValueType &register_sin,
                                    const ValueType &register_cos,
                                    remove_complex<ValueType> *residual_norm,
                                    ValueType *residual_norm_collection,
                                    size_type stride_residual_norm_collection)
{
    const auto this_rnc =
        residual_norm_collection[iter * stride_residual_norm_collection +
                                 col_idx];
    const auto next_rnc = -conj(register_sin) * this_rnc;
    residual_norm_collection[iter * stride_residual_norm_collection + col_idx] =
        register_cos * this_rnc;
    residual_norm[col_idx] = dpcpp::abs(next_rnc);
    residual_norm_collection[(iter + 1) * stride_residual_norm_collection +
                             col_idx] = next_rnc;
}


// Must be called with at least `num_cols` threads in total.
template <size_type block_size, typename ValueType>
void givens_rotation_kernel(
    size_type num_rows, size_type num_cols, size_type iter,
    ValueType *__restrict__ hessenberg_iter, size_type stride_hessenberg,
    ValueType *__restrict__ givens_sin, size_type stride_sin,
    ValueType *__restrict__ givens_cos, size_type stride_cos,
    remove_complex<ValueType> *__restrict__ residual_norm,
    ValueType *__restrict__ residual_norm_collection,
    size_type stride_residual_norm_collection,
    const stopping_status *__restrict__ stop_status, sycl::nd_item<3> item_ct1)
{
    const auto col_idx = thread::get_thread_id_flat(item_ct1);

    if (col_idx >= num_cols || stop_status[col_idx].has_stopped()) {
        return;
    }

    auto this_hess = hessenberg_iter[col_idx];
    auto next_hess = hessenberg_iter[stride_hessenberg + col_idx];
    for (size_type i = 0; i < iter; ++i) {
        const auto cos = givens_cos[i * stride_cos + col_idx];
        const auto sin = givens_sin[i * stride_sin + col_idx];
        hessenberg_iter[i * stride_hessenberg + col_idx] =
            cos * this_hess + sin * next_hess;
        this_hess = conj(cos) * next_hess - conj(sin) * this_hess;
        next_hess = hessenberg_iter[(i + 2) * stride_hessenberg + col_idx];
    }
    // for j in 0:iter - 1
    //     temp             =  cos(j)*hessenberg(j) +
    //                         sin(j)*hessenberg(j+1)
    //     hessenberg(j+1)  = -sin(j)*hessenberg(j) +
    //                         cos(j)*hessenberg(j+1)
    //     hessenberg(j)    =  temp;
    // end

    ValueType register_sin;
    ValueType register_cos;
    calculate_sin_and_cos_kernel(col_idx, num_cols, iter, this_hess, next_hess,
                                 givens_sin, stride_sin, givens_cos, stride_cos,
                                 register_sin, register_cos);
    // Calculate sin and cos on hessenberg(iter) and hessenberg(iter+1)

    hessenberg_iter[iter * stride_hessenberg + col_idx] =
        register_cos * this_hess + register_sin * next_hess;
    hessenberg_iter[(iter + 1) * stride_hessenberg + col_idx] =
        zero<ValueType>();
    // hessenberg(iter)   = cos(iter)*hessenberg(iter) +
    //                      sin(iter)*hessenberg(iter+1)
    // hessenberg(iter+1) = 0

    calculate_residual_norm_kernel(
        col_idx, num_cols, iter, register_sin, register_cos, residual_norm,
        residual_norm_collection, stride_residual_norm_collection);
    // Calculate residual norm
}

template <size_type block_size, typename ValueType>
void givens_rotation_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                            sycl::queue *stream, size_type num_rows,
                            size_type num_cols, size_type iter,
                            ValueType *hessenberg_iter,
                            size_type stride_hessenberg, ValueType *givens_sin,
                            size_type stride_sin, ValueType *givens_cos,
                            size_type stride_cos,
                            remove_complex<ValueType> *residual_norm,
                            ValueType *residual_norm_collection,
                            size_type stride_residual_norm_collection,
                            const stopping_status *stop_status)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item_ct1) {
                givens_rotation_kernel<block_size>(
                    num_rows, num_cols, iter, hessenberg_iter,
                    stride_hessenberg, givens_sin, stride_sin, givens_cos,
                    stride_cos, residual_norm, residual_norm_collection,
                    stride_residual_norm_collection, stop_status, item_ct1);
            });
    });
}


// Must be called with at least `num_rhs` threads in total.
template <size_type block_size, typename ValueType>
void solve_upper_triangular_kernel(
    size_type num_cols, size_type num_rhs,
    const ValueType *__restrict__ residual_norm_collection,
    size_type stride_residual_norm_collection,
    const ValueType *__restrict__ hessenberg, size_type stride_hessenberg,
    ValueType *__restrict__ y, size_type stride_y,
    const size_type *__restrict__ final_iter_nums, sycl::nd_item<3> item_ct1)
{
    const auto col_idx = thread::get_thread_id_flat(item_ct1);

    if (col_idx >= num_rhs) {
        return;
    }

    for (int i = final_iter_nums[col_idx] - 1; i >= 0; --i) {
        auto temp =
            residual_norm_collection[i * stride_residual_norm_collection +
                                     col_idx];
        for (size_type j = i + 1; j < final_iter_nums[col_idx]; ++j) {
            temp -= hessenberg[i * stride_hessenberg + j * num_rhs + col_idx] *
                    y[j * stride_y + col_idx];
        }

        y[i * stride_y + col_idx] =
            temp / hessenberg[i * stride_hessenberg + i * num_rhs + col_idx];
    }
    // Solve upper triangular.
    // y = hessenberg \ residual_norm_collection
}

template <size_type block_size, typename ValueType>
void solve_upper_triangular_kernel(
    dim3 grid, dim3 block, size_t dynamic_shared_memory, sycl::queue *stream,
    size_type num_cols, size_type num_rhs,
    const ValueType *residual_norm_collection,
    size_type stride_residual_norm_collection, const ValueType *hessenberg,
    size_type stride_hessenberg, ValueType *y, size_type stride_y,
    const size_type *final_iter_nums)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             solve_upper_triangular_kernel<block_size>(
                                 num_cols, num_rhs, residual_norm_collection,
                                 stride_residual_norm_collection, hessenberg,
                                 stride_hessenberg, y, stride_y,
                                 final_iter_nums, item_ct1);
                         });
    });
}


// Must be called with at least `stride_preconditioner * num_rows` threads in
// total.
template <size_type block_size, typename ValueType>
void calculate_Qy_kernel(size_type num_rows, size_type num_cols,
                         size_type num_rhs,
                         const ValueType *__restrict__ krylov_bases,
                         size_type stride_krylov,
                         const ValueType *__restrict__ y, size_type stride_y,
                         ValueType *__restrict__ before_preconditioner,
                         size_type stride_preconditioner,
                         const size_type *__restrict__ final_iter_nums,
                         sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_id = global_id / stride_preconditioner;
    const auto col_id = global_id % stride_preconditioner;

    if (row_id < num_rows && col_id < num_cols) {
        ValueType temp = zero<ValueType>();

        for (size_type j = 0; j < final_iter_nums[col_id]; ++j) {
            temp +=
                krylov_bases[(row_id + j * num_rows) * stride_krylov + col_id] *
                y[j * stride_y + col_id];
        }
        before_preconditioner[global_id] = temp;
    }
}

template <size_type block_size, typename ValueType>
void calculate_Qy_kernel(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                         sycl::queue *stream, size_type num_rows,
                         size_type num_cols, size_type num_rhs,
                         const ValueType *krylov_bases, size_type stride_krylov,
                         const ValueType *y, size_type stride_y,
                         ValueType *before_preconditioner,
                         size_type stride_preconditioner,
                         const size_type *final_iter_nums)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             calculate_Qy_kernel<block_size>(
                                 num_rows, num_cols, num_rhs, krylov_bases,
                                 stride_krylov, y, stride_y,
                                 before_preconditioner, stride_preconditioner,
                                 final_iter_nums, item_ct1);
                         });
    });
}


template <typename ValueType>
void initialize_1(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *givens_sin,
                  matrix::Dense<ValueType> *givens_cos,
                  Array<stopping_status> *stop_status, size_type krylov_dim)
{
    const auto num_threads = std::max(b->get_size()[0] * b->get_stride(),
                                      krylov_dim * b->get_size()[1]);
    const dim3 grid_dim(ceildiv(num_threads, default_block_size), 1, 1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;

    initialize_1_kernel<block_size>(
        grid_dim, block_dim, 0, exec->get_queue(), b->get_size()[0],
        b->get_size()[1], krylov_dim, b->get_const_values(), b->get_stride(),
        residual->get_values(), residual->get_stride(),
        givens_sin->get_values(), givens_sin->get_stride(),
        givens_cos->get_values(), givens_cos->get_stride(),
        stop_status->get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL);


template <typename ValueType>
void initialize_2(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<remove_complex<ValueType>> *residual_norm,
                  matrix::Dense<ValueType> *residual_norm_collection,
                  matrix::Dense<ValueType> *krylov_bases,
                  Array<size_type> *final_iter_nums, size_type krylov_dim)
{
    const auto num_rows = residual->get_size()[0];
    const auto num_rhs = residual->get_size()[1];
    const dim3 grid_dim_1(
        ceildiv(krylov_bases->get_size()[0] * krylov_bases->get_stride(),
                default_block_size),
        1, 1);
    const dim3 block_dim(default_block_size, 1, 1);
    constexpr auto block_size = default_block_size;

    residual->compute_norm2(residual_norm);

    const dim3 grid_dim_2(ceildiv(num_rows * num_rhs, default_block_size), 1,
                          1);
    initialize_2_2_kernel<block_size>(
        grid_dim_2, block_dim, 0, exec->get_queue(), residual->get_size()[0],
        residual->get_size()[1], residual->get_const_values(),
        residual->get_stride(), residual_norm->get_const_values(),
        residual_norm_collection->get_values(), krylov_bases->get_values(),
        krylov_bases->get_stride(), final_iter_nums->get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL);


template <typename ValueType>
void finish_arnoldi(std::shared_ptr<const DpcppExecutor> exec,
                    size_type num_rows, matrix::Dense<ValueType> *krylov_bases,
                    matrix::Dense<ValueType> *hessenberg_iter, size_type iter,
                    const stopping_status *stop_status)
{
    const auto stride_krylov = krylov_bases->get_stride();
    const auto stride_hessenberg = hessenberg_iter->get_stride();
    // auto cublas_handle = exec->get_cublas_handle();
    const dim3 grid_size(
        ceildiv(hessenberg_iter->get_size()[1], default_dot_dim),
        exec->get_num_computing_units() * 2);
    const dim3 block_size(default_dot_dim, default_dot_dim);
    auto next_krylov_basis =
        krylov_bases->get_values() +
        (iter + 1) * num_rows * hessenberg_iter->get_size()[1];
    for (size_type k = 0; k < iter + 1; ++k) {
        const auto k_krylov_bases =
            krylov_bases->get_const_values() +
            k * num_rows * hessenberg_iter->get_size()[1];
        if (hessenberg_iter->get_size()[1] >= 1) {
            // TODO: this condition should be tuned
            // single rhs will use vendor's dot, otherwise, use our own
            // multidot_kernel which parallelize multiple rhs.
            components::fill_array(
                exec, hessenberg_iter->get_values() + k * stride_hessenberg,
                hessenberg_iter->get_size()[1], zero<ValueType>());
            multidot_kernel(grid_size, block_size, 0, exec->get_queue(), k,
                            num_rows, hessenberg_iter->get_size()[1],
                            k_krylov_bases, next_krylov_basis, stride_krylov,
                            hessenberg_iter->get_values(), stride_hessenberg,
                            stop_status);
        } else {
            oneapi::mkl::blas::row_major::dot(
                *exec->get_queue(), num_rows, k_krylov_bases, stride_krylov,
                next_krylov_basis, stride_krylov,
                hessenberg_iter->get_values() + k * stride_hessenberg);
        }
        update_next_krylov_kernel<default_block_size>(
            ceildiv(num_rows * stride_krylov, default_block_size),
            default_block_size, 0, exec->get_queue(), k, num_rows,
            hessenberg_iter->get_size()[1], k_krylov_bases, next_krylov_basis,
            stride_krylov, hessenberg_iter->get_const_values(),
            stride_hessenberg, stop_status);
    }
    // for i in 1:iter
    //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
    //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
    // end


    update_hessenberg_2_kernel<default_block_size>(
        hessenberg_iter->get_size()[1], default_block_size, 0,
        exec->get_queue(), iter, num_rows, hessenberg_iter->get_size()[1],
        next_krylov_basis, stride_krylov, hessenberg_iter->get_values(),
        stride_hessenberg, stop_status);

    update_krylov_kernel<default_block_size>(
        ceildiv(num_rows * stride_krylov, default_block_size),
        default_block_size, 0, exec->get_queue(), iter, num_rows,
        hessenberg_iter->get_size()[1], next_krylov_basis, stride_krylov,
        hessenberg_iter->get_const_values(), stride_hessenberg, stop_status);
    // next_krylov_basis /= hessenberg(iter, iter + 1)
    // End of arnoldi
}


template <typename ValueType>
void givens_rotation(std::shared_ptr<const DpcppExecutor> exec,
                     matrix::Dense<ValueType> *givens_sin,
                     matrix::Dense<ValueType> *givens_cos,
                     matrix::Dense<ValueType> *hessenberg_iter,
                     matrix::Dense<remove_complex<ValueType>> *residual_norm,
                     matrix::Dense<ValueType> *residual_norm_collection,
                     size_type iter, const Array<stopping_status> *stop_status)
{
    // TODO: tune block_size for optimal performance
    constexpr auto block_size = default_block_size;
    const auto num_cols = hessenberg_iter->get_size()[1];
    const dim3 block_dim{block_size, 1, 1};
    const dim3 grid_dim{
        static_cast<unsigned int>(ceildiv(num_cols, block_size)), 1, 1};

    givens_rotation_kernel<block_size>(
        grid_dim, block_dim, 0, exec->get_queue(),
        hessenberg_iter->get_size()[0], hessenberg_iter->get_size()[1], iter,
        hessenberg_iter->get_values(), hessenberg_iter->get_stride(),
        givens_sin->get_values(), givens_sin->get_stride(),
        givens_cos->get_values(), givens_cos->get_stride(),
        residual_norm->get_values(), residual_norm_collection->get_values(),
        residual_norm_collection->get_stride(), stop_status->get_const_data());
}


template <typename ValueType>
void step_1(std::shared_ptr<const DpcppExecutor> exec, size_type num_rows,
            matrix::Dense<ValueType> *givens_sin,
            matrix::Dense<ValueType> *givens_cos,
            matrix::Dense<remove_complex<ValueType>> *residual_norm,
            matrix::Dense<ValueType> *residual_norm_collection,
            matrix::Dense<ValueType> *krylov_bases,
            matrix::Dense<ValueType> *hessenberg_iter, size_type iter,
            Array<size_type> *final_iter_nums,
            const Array<stopping_status> *stop_status)
{
    increase_final_iteration_numbers_kernel(
        static_cast<unsigned int>(
            ceildiv(final_iter_nums->get_num_elems(), default_block_size)),
        default_block_size, 0, exec->get_queue(), final_iter_nums->get_data(),
        stop_status->get_const_data(), final_iter_nums->get_num_elems());
    finish_arnoldi(exec, num_rows, krylov_bases, hessenberg_iter, iter,
                   stop_status->get_const_data());
    givens_rotation(exec, givens_sin, givens_cos, hessenberg_iter,
                    residual_norm, residual_norm_collection, iter, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_1_KERNEL);


template <typename ValueType>
void solve_upper_triangular(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Dense<ValueType> *residual_norm_collection,
    const matrix::Dense<ValueType> *hessenberg, matrix::Dense<ValueType> *y,
    const Array<size_type> *final_iter_nums)
{
    // TODO: tune block_size for optimal performance
    constexpr auto block_size = default_block_size;
    const auto num_rhs = residual_norm_collection->get_size()[1];
    const dim3 block_dim{block_size, 1, 1};
    const dim3 grid_dim{static_cast<unsigned int>(ceildiv(num_rhs, block_size)),
                        1, 1};

    solve_upper_triangular_kernel<block_size>(
        grid_dim, block_dim, 0, exec->get_queue(), hessenberg->get_size()[1],
        num_rhs, residual_norm_collection->get_const_values(),
        residual_norm_collection->get_stride(), hessenberg->get_const_values(),
        hessenberg->get_stride(), y->get_values(), y->get_stride(),
        final_iter_nums->get_const_data());
}


template <typename ValueType>
void calculate_qy(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::Dense<ValueType> *krylov_bases,
                  const matrix::Dense<ValueType> *y,
                  matrix::Dense<ValueType> *before_preconditioner,
                  const Array<size_type> *final_iter_nums)
{
    const auto num_rows = before_preconditioner->get_size()[0];
    const auto num_cols = krylov_bases->get_size()[1];
    const auto num_rhs = before_preconditioner->get_size()[1];
    const auto stride_before_preconditioner =
        before_preconditioner->get_stride();

    constexpr auto block_size = default_block_size;
    const dim3 grid_dim{
        static_cast<unsigned int>(
            ceildiv(num_rows * stride_before_preconditioner, block_size)),
        1, 1};
    const dim3 block_dim{block_size, 1, 1};


    calculate_Qy_kernel<block_size>(
        grid_dim, block_dim, 0, exec->get_queue(), num_rows, num_cols, num_rhs,
        krylov_bases->get_const_values(), krylov_bases->get_stride(),
        y->get_const_values(), y->get_stride(),
        before_preconditioner->get_values(), stride_before_preconditioner,
        final_iter_nums->get_const_data());
    // Calculate qy
    // before_preconditioner = krylov_bases * y
}


template <typename ValueType>
void step_2(std::shared_ptr<const DpcppExecutor> exec,
            const matrix::Dense<ValueType> *residual_norm_collection,
            const matrix::Dense<ValueType> *krylov_bases,
            const matrix::Dense<ValueType> *hessenberg,
            matrix::Dense<ValueType> *y,
            matrix::Dense<ValueType> *before_preconditioner,
            const Array<size_type> *final_iter_nums)
{
    solve_upper_triangular(exec, residual_norm_collection, hessenberg, y,
                           final_iter_nums);
    calculate_qy(exec, krylov_bases, y, before_preconditioner, final_iter_nums);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_2_KERNEL);


}  // namespace gmres
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
