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

#include <ginkgo/core/matrix/ell.hpp>


#include <algorithm>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/allocator.hpp"
#include "core/components/absolute_array_kernels.hpp"
#include "core/components/device_matrix_data_kernels.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"


namespace gko {
namespace matrix {
namespace ell {
namespace {


GKO_REGISTER_OPERATION(spmv, ell::spmv);
GKO_REGISTER_OPERATION(advanced_spmv, ell::advanced_spmv);
GKO_REGISTER_OPERATION(build_row_ptrs, components::build_row_ptrs);
GKO_REGISTER_OPERATION(compute_max_row_nnz, ell::compute_max_row_nnz);
GKO_REGISTER_OPERATION(fill_in_matrix_data, ell::fill_in_matrix_data);
GKO_REGISTER_OPERATION(convert_to_dense, ell::convert_to_dense);
GKO_REGISTER_OPERATION(convert_to_csr, ell::convert_to_csr);
GKO_REGISTER_OPERATION(count_nonzeros, ell::count_nonzeros);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row,
                       ell::calculate_nonzeros_per_row);
GKO_REGISTER_OPERATION(extract_diagonal, ell::extract_diagonal);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(inplace_absolute_array,
                       components::inplace_absolute_array);
GKO_REGISTER_OPERATION(outplace_absolute_array,
                       components::outplace_absolute_array);


}  // anonymous namespace
}  // namespace ell


namespace {


template <typename ValueType, typename IndexType>
size_type calculate_max_nnz_per_row(
    const matrix_data<ValueType, IndexType>& data)
{
    size_type nnz = 0;
    IndexType current_row = 0;
    size_type num_stored_elements_per_row = 0;
    for (const auto& elem : data.nonzeros) {
        if (elem.row != current_row) {
            current_row = elem.row;
            num_stored_elements_per_row =
                std::max(num_stored_elements_per_row, nnz);
            nnz = 0;
        }
        nnz += (elem.value != zero<ValueType>());
    }
    return std::max(num_stored_elements_per_row, nnz);
}


}  // namespace


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    mixed_precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->get_executor()->run(ell::make_spmv(this, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                           const LinOp* beta, LinOp* x) const
{
    mixed_precision_dispatch_real_complex<ValueType>(
        [this, alpha, beta](auto dense_b, auto dense_x) {
            auto dense_alpha = make_temporary_conversion<ValueType>(alpha);
            auto dense_beta = make_temporary_conversion<
                typename std::decay_t<decltype(*dense_x)>::value_type>(beta);
            this->get_executor()->run(ell::make_advanced_spmv(
                dense_alpha.get(), this, dense_b, dense_beta.get(), dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(
    Ell<next_precision<ValueType>, IndexType>* result) const
{
    result->values_ = this->values_;
    result->col_idxs_ = this->col_idxs_;
    result->num_stored_elements_per_row_ = this->num_stored_elements_per_row_;
    result->stride_ = this->stride_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::move_to(
    Ell<next_precision<ValueType>, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(Dense<ValueType>* result) const
{
    auto exec = this->get_executor();
    auto tmp = Dense<ValueType>::create(exec, this->get_size());
    exec->run(ell::make_convert_to_dense(this, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::move_to(Dense<ValueType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::convert_to(
    Csr<ValueType, IndexType>* result) const
{
    auto exec = this->get_executor();

    size_type num_stored_elements = 0;
    exec->run(ell::make_count_nonzeros(this, &num_stored_elements));

    auto tmp = Csr<ValueType, IndexType>::create(
        exec, this->get_size(), num_stored_elements, result->get_strategy());
    exec->run(ell::make_convert_to_csr(this, tmp.get()));

    tmp->make_srow();
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::move_to(Csr<ValueType, IndexType>* result)
{
    this->convert_to(result);
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::read(const device_mat_data& data)
{
    auto exec = this->get_executor();
    Array<int64> row_ptrs(exec, data.size[0] + 1);
    auto local_data = make_temporary_clone(exec, &data.nonzeros);
    exec->run(ell::make_build_row_ptrs(*local_data, data.size[0],
                                       row_ptrs.get_data()));
    size_type max_nnz{};
    exec->run(ell::make_compute_max_row_nnz(row_ptrs, max_nnz));
    if (this->get_size() != data.size ||
        this->get_num_stored_elements_per_row() != max_nnz) {
        stride_ = data.size[0];
        values_.resize_and_reset(stride_ * max_nnz);
        col_idxs_.resize_and_reset(stride_ * max_nnz);
        num_stored_elements_per_row_ = max_nnz;
        this->set_size(data.size);
    }
    exec->run(ell::make_fill_in_matrix_data(*local_data,
                                            row_ptrs.get_const_data(), this));
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::read(const mat_data& data)
{
    this->read(device_mat_data::create_view_from_host(
        this->get_executor(), const_cast<mat_data&>(data)));
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::write(mat_data& data) const
{
    std::unique_ptr<const LinOp> op{};
    const Ell* tmp{};
    if (this->get_executor()->get_master() != this->get_executor()) {
        op = this->clone(this->get_executor()->get_master());
        tmp = static_cast<const Ell*>(op.get());
    } else {
        tmp = this;
    }

    data = {tmp->get_size(), {}};

    for (size_type row = 0; row < tmp->get_size()[0]; ++row) {
        for (size_type i = 0; i < tmp->num_stored_elements_per_row_; ++i) {
            const auto val = tmp->val_at(row, i);
            if (val != zero<ValueType>()) {
                const auto col = tmp->col_at(row, i);
                data.nonzeros.emplace_back(row, col, val);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Diagonal<ValueType>>
Ell<ValueType, IndexType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(ell::make_fill_array(diag->get_values(), diag->get_size()[0],
                                   zero<ValueType>()));
    exec->run(ell::make_extract_diagonal(this, lend(diag)));
    return diag;
}


template <typename ValueType, typename IndexType>
void Ell<ValueType, IndexType>::compute_absolute_inplace()
{
    auto exec = this->get_executor();

    exec->run(ell::make_inplace_absolute_array(
        this->get_values(), this->get_num_stored_elements()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<typename Ell<ValueType, IndexType>::absolute_type>
Ell<ValueType, IndexType>::compute_absolute() const
{
    auto exec = this->get_executor();

    auto abs_ell = absolute_type::create(
        exec, this->get_size(), this->get_num_stored_elements_per_row(),
        this->get_stride());

    abs_ell->col_idxs_ = col_idxs_;
    exec->run(ell::make_outplace_absolute_array(this->get_const_values(),
                                                this->get_num_stored_elements(),
                                                abs_ell->get_values()));

    return abs_ell;
}


#define GKO_DECLARE_ELL_MATRIX(ValueType, IndexType) \
    class Ell<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_MATRIX);


}  // namespace matrix
}  // namespace gko
