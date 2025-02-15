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

#ifndef GKO_CORE_COMPONENTS_DEVICE_MATRIX_DATA_KERNELS_HPP_
#define GKO_CORE_COMPONENTS_DEVICE_MATRIX_DATA_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_DEVICE_MATRIX_DATA_REMOVE_ZEROS_KERNEL(ValueType, \
                                                           IndexType) \
    void remove_zeros(std::shared_ptr<const DefaultExecutor> exec,    \
                      Array<matrix_data_entry<ValueType, IndexType>>& data)

#define GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL(ValueType, \
                                                             IndexType) \
    void sort_row_major(std::shared_ptr<const DefaultExecutor> exec,    \
                        Array<matrix_data_entry<ValueType, IndexType>>& data)

#define GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_KERNEL(       \
    ValueType, IndexType, RowPtrType)                               \
    void build_row_ptrs(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                \
        const Array<matrix_data_entry<ValueType, IndexType>>& data, \
        size_type num_rows, RowPtrType* row_ptrs)

#define GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_KERNEL32(ValueType,      \
                                                               IndexType)      \
    GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_KERNEL(ValueType, IndexType, \
                                                         ::gko::int32)
#define GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_KERNEL64(ValueType,      \
                                                               IndexType)      \
    GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_KERNEL(ValueType, IndexType, \
                                                         ::gko::int64)

#define GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_FROM_IDXS_KERNEL(        \
    IndexType, RowPtrType)                                                     \
    void build_row_ptrs_from_idxs(std::shared_ptr<const DefaultExecutor> exec, \
                                  const Array<IndexType>& row_idxs,            \
                                  size_type num_rows, RowPtrType* row_ptrs)

#define GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_FROM_IDXS_KERNEL32( \
    IndexType)                                                            \
    GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_FROM_IDXS_KERNEL(       \
        IndexType, ::gko::int32)
#define GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_FROM_IDXS_KERNEL64( \
    IndexType)                                                            \
    GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_FROM_IDXS_KERNEL(       \
        IndexType, ::gko::int64)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                           \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_DEVICE_MATRIX_DATA_REMOVE_ZEROS_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_DEVICE_MATRIX_DATA_SORT_ROW_MAJOR_KERNEL(ValueType,            \
                                                         IndexType);           \
    template <typename ValueType, typename IndexType, typename RowPtrType>     \
    GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_KERNEL(ValueType, IndexType, \
                                                         RowPtrType);          \
    template <typename IndexType, typename RowPtrType>                         \
    GKO_DECLARE_DEVICE_MATRIX_DATA_BUILD_ROW_PTRS_FROM_IDXS_KERNEL(IndexType,  \
                                                                   RowPtrType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(components,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_DEVICE_MATRIX_DATA_KERNELS_HPP_
