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

#ifndef GKO_PUBLIC_KERNELS_KERNEL_LAUNCH_HPP_
#define GKO_PUBLIC_KERNELS_KERNEL_LAUNCH_HPP_


#include <memory>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#if defined(GKO_COMPILING_CUDA)

#define GKO_DEVICE_NAMESPACE cuda
#define GKO_KERNEL [] __device__
#include <ginkgo/kernels/cuda/types.hpp>


namespace gko {
namespace kernels {
namespace cuda {


template <typename T>
using device_type = typename detail::cuda_type_impl<T>::type;

template <typename T>
device_type<T> as_device_type(T value)
{
    return as_cuda_type(value);
}


template <typename T>
using unpack_member_type = typename detail::fake_complex_unpack_impl<T>::type;

template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr unpack_member_type<T> unpack_member(T value)
{
    return fake_complex_unpack(value);
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#elif defined(GKO_COMPILING_HIP)

#define GKO_DEVICE_NAMESPACE hip
#define GKO_KERNEL [] __device__
#include <ginkgo/kernels/hip/types.hip.hpp>


namespace gko {
namespace kernels {
namespace hip {


template <typename T>
using device_type = typename detail::hip_type_impl<T>::type;

template <typename T>
device_type<T> as_device_type(T value)
{
    static_assert(sizeof(device_type<T>) == sizeof(T), "Mapping changes size");
    return as_hip_type(value);
}


template <typename T>
using unpack_member_type = typename detail::fake_complex_unpack_impl<T>::type;

template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr unpack_member_type<T> unpack_member(T value)
{
    return fake_complex_unpack(value);
}


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#elif defined(GKO_COMPILING_DPCPP)

#define GKO_DEVICE_NAMESPACE dpcpp
#define GKO_KERNEL []


namespace gko {
namespace kernels {
namespace dpcpp {


template <typename T>
using device_type = T;

template <typename T>
device_type<T> as_device_type(T value)
{
    return value;
}


template <typename T>
using unpack_member_type = T;

template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr unpack_member_type<T> unpack_member(T value)
{
    return value;
}

}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#elif defined(GKO_COMPILING_OMP)

#define GKO_DEVICE_NAMESPACE omp
#define GKO_KERNEL []


namespace gko {
namespace kernels {
namespace omp {


template <typename T>
using device_type = T;

template <typename T>
device_type<T> as_device_type(T value)
{
    return value;
}


template <typename T>
using unpack_member_type = T;

template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr unpack_member_type<T> unpack_member(T value)
{
    return value;
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko


#elif defined(GKO_COMPILING_REFERENCE)

#define GKO_DEVICE_NAMESPACE reference
#define GKO_KERNEL []


namespace gko {
namespace kernels {
namespace reference {


template <typename T>
using device_type = T;

template <typename T>
device_type<T> as_device_type(T value)
{
    return value;
}


template <typename T>
using unpack_member_type = T;

template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr unpack_member_type<T> unpack_member(T value)
{
    return value;
}


}  // namespace reference
}  // namespace kernels
}  // namespace gko


#elif defined(GKO_COMPILING_STUB)


#define GKO_KERNEL []


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


template <typename T>
using device_type = T;

template <typename T>
device_type<T> as_device_type(T value)
{
    return value;
}


template <typename T>
using unpack_member_type = T;

template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr unpack_member_type<T> unpack_member(T value)
{
    return value;
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#else

#error "This file should only be used inside Ginkgo device compilation"

#endif


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


/**
 * @internal
 * A simple row-major accessor as a device representation of gko::matrix::Dense
 * objects.
 *
 * @tparam ValueType  the value type of the underlying matrix.
 */
template <typename ValueType>
struct matrix_accessor {
    ValueType* data;
    int64 stride;

    /**
     * @internal
     * Returns a reference to the element at position (row, col).
     */
    GKO_INLINE GKO_ATTRIBUTES ValueType& operator()(int64 row, int64 col)
    {
        return data[row * stride + col];
    }

    /**
     * @internal
     * Returns a reference to the element at position idx in the underlying
     * storage.
     */
    GKO_INLINE GKO_ATTRIBUTES ValueType& operator[](int64 idx)
    {
        return data[idx];
    }
};


/**
 * @internal
 * This struct is used to provide mappings from host types like
 * gko::matrix::Dense to device representations of the same data, like an
 * accessor storing only data pointer and stride.
 *
 * By default, it only maps std::complex to the corresponding device
 * representation of the complex type. There are specializations for dealing
 * with gko::Array and gko::matrix::Dense (both const and mutable) that map them
 * to plain pointers or matrix_accessor objects.
 *
 * @tparam T  the type being mapped. It will be used based on a
 *            forwarding-reference, i.e. preserve references in the input
 *            parameter, so special care must be taken to only return types that
 *            can be passed to the device, i.e. (structs containing) device
 *            pointers or values. This means that T will be either a r-value or
 *            l-value reference.
 */
template <typename T>
struct to_device_type_impl {
    using type = std::decay_t<device_type<T>>;
    static type map_to_device(T in) { return as_device_type(in); }
};

template <typename T>
struct to_device_type_impl<std::shared_ptr<T>&> {
    using type = typename to_device_type_impl<T*&>::type;
    static type map_to_device(std::shared_ptr<T>& in)
    {
        return to_device_type_impl<T*&>::map_to_device(in.get());
    }
};

template <typename T>
struct to_device_type_impl<const std::shared_ptr<T>&> {
    using type = typename to_device_type_impl<T*&>::type;
    static type map_to_device(const std::shared_ptr<T>& in)
    {
        return to_device_type_impl<T*&>::map_to_device(in.get());
    }
};

template <typename T>
struct to_device_type_impl<std::unique_ptr<T>&> {
    using type = typename to_device_type_impl<T*&>::type;
    static type map_to_device(std::unique_ptr<T>& in)
    {
        return to_device_type_impl<T*&>::map_to_device(in.get());
    }
};

template <typename T>
struct to_device_type_impl<const std::unique_ptr<T>&> {
    using type = typename to_device_type_impl<T*&>::type;
    static type map_to_device(const std::unique_ptr<T>& in)
    {
        return to_device_type_impl<T*&>::map_to_device(in.get());
    }
};

template <typename ValueType>
struct to_device_type_impl<matrix::Dense<ValueType>*&> {
    using type = matrix_accessor<device_type<ValueType>>;
    static type map_to_device(matrix::Dense<ValueType>* mtx)
    {
        return {as_device_type(mtx->get_values()),
                static_cast<int64>(mtx->get_stride())};
    }
};

template <typename ValueType>
struct to_device_type_impl<const matrix::Dense<ValueType>*&> {
    using type = matrix_accessor<const device_type<ValueType>>;
    static type map_to_device(const matrix::Dense<ValueType>* mtx)
    {
        return {as_device_type(mtx->get_const_values()),
                static_cast<int64>(mtx->get_stride())};
    }
};

template <typename ValueType>
struct to_device_type_impl<Array<ValueType>&> {
    using type = device_type<ValueType>*;
    static type map_to_device(Array<ValueType>& array)
    {
        return as_device_type(array.get_data());
    }
};

template <typename ValueType>
struct to_device_type_impl<const Array<ValueType>&> {
    using type = const device_type<ValueType>*;
    static type map_to_device(const Array<ValueType>& array)
    {
        return as_device_type(array.get_const_data());
    }
};


template <typename T>
typename to_device_type_impl<T>::type map_to_device(T&& param)
{
    return to_device_type_impl<T>::map_to_device(param);
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#if defined(GKO_COMPILING_CUDA)
#include <ginkgo/kernels/cuda/kernel_launch.cuh>
#elif defined(GKO_COMPILING_HIP)
#include <ginkgo/kernels/hip/kernel_launch.hip.hpp>
#elif defined(GKO_COMPILING_DPCPP)
#include <ginkgo/kernels/dpcpp/kernel_launch.dp.hpp>
#elif defined(GKO_COMPILING_OMP)
#include <ginkgo/kernels/omp/kernel_launch.hpp>
#elif defined(GKO_COMPILING_REFERENCE)
#include <ginkgo/kernels/reference/kernel_launch.hpp>
#elif defined(GKO_COMPILING_STUB)
#include <ginkgo/kernels/stub/kernel_launch.hpp>
#endif


#endif  // GKO_PUBLIC_KERNELS_KERNEL_LAUNCH_HPP_
