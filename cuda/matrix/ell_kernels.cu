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

#include "core/matrix/ell_kernels.hpp"


#include <array>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/synthesizer/implementation_selection.hpp>
#include <ginkgo/kernels/cuda/types.hpp>


#include "accessor/reduced_row_major.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/format_conversion.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The ELL matrix format namespace.
 *
 * @ingroup ell
 */
namespace ell {


constexpr int default_block_size = 512;


// TODO: num_threads_per_core and ratio are parameters should be tuned
/**
 * num_threads_per_core is the oversubscribing parameter. There are
 * `num_threads_per_core` threads assigned to each physical core.
 */
constexpr int num_threads_per_core = 4;


/**
 * ratio is the parameter to decide when to use threads to do reduction on each
 * row. (#cols/#rows > ratio)
 */
constexpr double ratio = 1e-2;


/**
 * max_thread_per_worker is the max number of thread per worker. The
 * `compiled_kernels` must be a list <0, 1, 2, ..., max_thread_per_worker>
 */
constexpr int max_thread_per_worker = 32;


/**
 * A compile-time list of sub-warp sizes for which the spmv kernels should be
 * compiled.
 * 0 is a special case where it uses a sub-warp size of warp_size in
 * combination with atomic_adds.
 */
using compiled_kernels = syn::value_list<int, 0, 1, 2, 4, 8, 16, 32>;


#include "common/cuda_hip/matrix/ell_kernels.hpp.inc"


namespace {

template <int dim, typename Type1, typename Type2>
GKO_INLINE auto as_cuda_accessor(
    const acc::range<acc::reduced_row_major<dim, Type1, Type2>>& acc)
{
    return acc::range<
        acc::reduced_row_major<dim, cuda_type<Type1>, cuda_type<Type2>>>(
        acc.get_accessor().get_size(),
        as_cuda_type(acc.get_accessor().get_stored_data()),
        acc.get_accessor().get_stride());
}


template <int info, typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void abstract_spmv(syn::value_list<int, info>, int num_worker_per_row,
                   const matrix::Ell<MatrixValueType, IndexType>* a,
                   const matrix::Dense<InputValueType>* b,
                   matrix::Dense<OutputValueType>* c,
                   const matrix::Dense<MatrixValueType>* alpha = nullptr,
                   const matrix::Dense<OutputValueType>* beta = nullptr)
{
    using a_accessor =
        gko::acc::reduced_row_major<1, OutputValueType, const MatrixValueType>;
    using b_accessor =
        gko::acc::reduced_row_major<2, OutputValueType, const InputValueType>;

    const auto nrows = a->get_size()[0];
    const auto stride = a->get_stride();
    const auto num_stored_elements_per_row =
        a->get_num_stored_elements_per_row();

    constexpr int num_thread_per_worker =
        (info == 0) ? max_thread_per_worker : info;
    constexpr bool atomic = (info == 0);
    const dim3 block_size(default_block_size / num_thread_per_worker,
                          num_thread_per_worker, 1);
    const dim3 grid_size(ceildiv(nrows * num_worker_per_row, block_size.x),
                         b->get_size()[1], 1);

    const auto a_vals = gko::acc::range<a_accessor>(
        std::array<size_type, 1>{{num_stored_elements_per_row * stride}},
        a->get_const_values());
    const auto b_vals = gko::acc::range<b_accessor>(
        std::array<size_type, 2>{{b->get_size()[0], b->get_size()[1]}},
        b->get_const_values(), std::array<size_type, 1>{{b->get_stride()}});

    if (alpha == nullptr && beta == nullptr) {
        kernel::spmv<num_thread_per_worker, atomic>
            <<<grid_size, block_size, 0, 0>>>(
                nrows, num_worker_per_row, as_cuda_accessor(a_vals),
                a->get_const_col_idxs(), stride, num_stored_elements_per_row,
                as_cuda_accessor(b_vals), as_cuda_type(c->get_values()),
                c->get_stride());
    } else if (alpha != nullptr && beta != nullptr) {
        const auto alpha_val = gko::acc::range<a_accessor>(
            std::array<size_type, 1>{1}, alpha->get_const_values());
        kernel::spmv<num_thread_per_worker, atomic>
            <<<grid_size, block_size, 0, 0>>>(
                nrows, num_worker_per_row, as_cuda_accessor(alpha_val),
                as_cuda_accessor(a_vals), a->get_const_col_idxs(), stride,
                num_stored_elements_per_row, as_cuda_accessor(b_vals),
                as_cuda_type(beta->get_const_values()),
                as_cuda_type(c->get_values()), c->get_stride());
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_abstract_spmv, abstract_spmv);


template <typename ValueType, typename IndexType>
std::array<int, 3> compute_thread_worker_and_atomicity(
    std::shared_ptr<const CudaExecutor> exec,
    const matrix::Ell<ValueType, IndexType>* a)
{
    int num_thread_per_worker = 1;
    int atomic = 0;
    int num_worker_per_row = 1;

    const auto nrows = a->get_size()[0];
    const auto ell_ncols = a->get_num_stored_elements_per_row();
    // TODO: num_threads_per_core should be tuned for AMD gpu
    const auto nwarps = exec->get_num_warps_per_sm() *
                        exec->get_num_multiprocessor() * num_threads_per_core;

    // Use multithreads to perform the reduction on each row when the matrix is
    // wide.
    // To make every thread have computation, so pick the value which is the
    // power of 2 less than max_thread_per_worker and is less than or equal to
    // ell_ncols. If the num_thread_per_worker is max_thread_per_worker and
    // allow more than one worker to work on the same row, use atomic add to
    // handle the worker write the value into the same position. The #worker is
    // decided according to the number of worker allowed on GPU.
    if (static_cast<double>(ell_ncols) / nrows > ratio) {
        while (num_thread_per_worker < max_thread_per_worker &&
               (num_thread_per_worker << 1) <= ell_ncols) {
            num_thread_per_worker <<= 1;
        }
        if (num_thread_per_worker == max_thread_per_worker) {
            num_worker_per_row =
                std::min(ell_ncols / max_thread_per_worker, nwarps / nrows);
            num_worker_per_row = std::max(num_worker_per_row, 1);
        }
        if (num_worker_per_row > 1) {
            atomic = 1;
        }
    }
    return {num_thread_per_worker, atomic, num_worker_per_row};
}


}  // namespace


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Ell<MatrixValueType, IndexType>* a,
          const matrix::Dense<InputValueType>* b,
          matrix::Dense<OutputValueType>* c)
{
    const auto data = compute_thread_worker_and_atomicity(exec, a);
    const int num_thread_per_worker = std::get<0>(data);
    const int atomic = std::get<1>(data);
    const int num_worker_per_row = std::get<2>(data);

    /**
     * info is the parameter for selecting the cuda kernel.
     * for info == 0, it uses the kernel by warp_size threads with atomic
     * operation for other value, it uses the kernel without atomic_add
     */
    const int info = (!atomic) * num_thread_per_worker;
    if (atomic) {
        components::fill_array(exec, c->get_values(),
                               c->get_num_stored_elements(),
                               zero<OutputValueType>());
    }
    select_abstract_spmv(
        compiled_kernels(),
        [&info](int compiled_info) { return info == compiled_info; },
        syn::value_list<int>(), syn::type_list<>(), num_worker_per_row, a, b,
        c);
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_SPMV_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<MatrixValueType>* alpha,
                   const matrix::Ell<MatrixValueType, IndexType>* a,
                   const matrix::Dense<InputValueType>* b,
                   const matrix::Dense<OutputValueType>* beta,
                   matrix::Dense<OutputValueType>* c)
{
    const auto data = compute_thread_worker_and_atomicity(exec, a);
    const int num_thread_per_worker = std::get<0>(data);
    const int atomic = std::get<1>(data);
    const int num_worker_per_row = std::get<2>(data);

    /**
     * info is the parameter for selecting the cuda kernel.
     * for info == 0, it uses the kernel by warp_size threads with atomic
     * operation for other value, it uses the kernel without atomic_add
     */
    const int info = (!atomic) * num_thread_per_worker;
    if (atomic) {
        dense::scale(exec, beta, c);
    }
    select_abstract_spmv(
        compiled_kernels(),
        [&info](int compiled_info) { return info == compiled_info; },
        syn::value_list<int>(), syn::type_list<>(), num_worker_per_row, a, b, c,
        alpha, beta);
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const CudaExecutor> exec,
                      const matrix::Ell<ValueType, IndexType>* source,
                      matrix::Dense<ValueType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto result_stride = result->get_stride();
    const auto col_idxs = source->get_const_col_idxs();
    const auto vals = source->get_const_values();
    const auto source_stride = source->get_stride();

    const dim3 block_size(config::warp_size,
                          config::max_block_size / config::warp_size, 1);
    const dim3 init_grid_dim(ceildiv(num_cols, block_size.x),
                             ceildiv(num_rows, block_size.y), 1);
    kernel::initialize_zero_dense<<<init_grid_dim, block_size>>>(
        num_rows, num_cols, result_stride, as_cuda_type(result->get_values()));

    const auto grid_dim = ceildiv(num_rows, default_block_size);
    kernel::fill_in_dense<<<grid_dim, default_block_size>>>(
        num_rows, source->get_num_stored_elements_per_row(), source_stride,
        as_cuda_type(col_idxs), as_cuda_type(vals), result_stride,
        as_cuda_type(result->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Ell<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    const auto stride = source->get_stride();
    const auto max_nnz_per_row = source->get_num_stored_elements_per_row();

    constexpr auto rows_per_block =
        ceildiv(default_block_size, config::warp_size);
    const auto grid_dim_nnz = ceildiv(source->get_size()[0], rows_per_block);

    kernel::count_nnz_per_row<<<grid_dim_nnz, default_block_size>>>(
        num_rows, max_nnz_per_row, stride,
        as_cuda_type(source->get_const_values()), as_cuda_type(row_ptrs));

    components::prefix_sum(exec, row_ptrs, num_rows + 1);

    size_type grid_dim = ceildiv(num_rows, default_block_size);

    kernel::fill_in_csr<<<grid_dim, default_block_size>>>(
        num_rows, max_nnz_per_row, stride,
        as_cuda_type(source->get_const_values()),
        as_cuda_type(source->get_const_col_idxs()), as_cuda_type(row_ptrs),
        as_cuda_type(col_idxs), as_cuda_type(values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Ell<ValueType, IndexType>* source,
                    size_type* result)
{
    const auto num_rows = source->get_size()[0];
    auto nnz_per_row = Array<size_type>(exec, num_rows);

    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    *result = reduce_add_array(exec, num_rows, nnz_per_row.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_COUNT_NONZEROS_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(std::shared_ptr<const CudaExecutor> exec,
                                const matrix::Ell<ValueType, IndexType>* source,
                                Array<size_type>* result)
{
    const auto num_rows = source->get_size()[0];
    const auto max_nnz_per_row = source->get_num_stored_elements_per_row();
    const auto stride = source->get_stride();
    const auto values = source->get_const_values();

    const auto warp_size = config::warp_size;
    const auto grid_dim = ceildiv(num_rows * warp_size, default_block_size);

    kernel::count_nnz_per_row<<<grid_dim, default_block_size>>>(
        num_rows, max_nnz_per_row, stride, as_cuda_type(values),
        as_cuda_type(result->get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const CudaExecutor> exec,
                      const matrix::Ell<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    const auto max_nnz_per_row = orig->get_num_stored_elements_per_row();
    const auto orig_stride = orig->get_stride();
    const auto diag_size = diag->get_size()[0];
    const auto num_blocks =
        ceildiv(diag_size * max_nnz_per_row, default_block_size);

    const auto orig_values = orig->get_const_values();
    const auto orig_col_idxs = orig->get_const_col_idxs();
    auto diag_values = diag->get_values();

    kernel::extract_diagonal<<<num_blocks, default_block_size>>>(
        diag_size, max_nnz_per_row, orig_stride, as_cuda_type(orig_values),
        as_cuda_type(orig_col_idxs), as_cuda_type(diag_values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_EXTRACT_DIAGONAL_KERNEL);


}  // namespace ell
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
