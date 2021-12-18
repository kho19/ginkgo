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

#include <ginkgo/ginkgo.hpp>
#include <ginkgo/kernels/kernel_launch.hpp>


namespace GKO_DEVICE_NAMESPACE {


using namespace gko::kernels::GKO_DEVICE_NAMESPACE;

template <typename T>
struct err {};

// explicit local update of the non-linear terms
void nonlinear_update(std::shared_ptr<const DefaultExecutor> exec, int nelems,
                      double tau, double f, double k, gko::matrix::Dense<>* x)
{
    run_kernel(
        exec,
        GKO_KERNEL(auto i, auto j, auto nelems, auto tau, auto f, auto k,
                   auto x) {
            if (i < nelems) {
                x[i] += tau * (-x[i] * (x[nelems + i] * x[nelems + i] + f) + f);
            } else {
                x[i] += tau * (x[i] * (x[i - nelems] * x[i] - (f + k)));
            }
        },
        gko::dim<2>{2 * nelems, 1}, nelems, tau, f, k, x);
}

}  // namespace GKO_DEVICE_NAMESPACE
