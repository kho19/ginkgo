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

//
// Created by heisenberg on 10.08.21.
//

// TODO: delete helper file when done

#pragma once

using mtx = gko::matrix::Csr<>;
using dense_mtx = gko::matrix::Dense<>;

/// Debugging funtions
inline void print_mat(std::shared_ptr<dense_mtx>& mat)
{
    std::cout << "\n";
    for (int i = 0; i < mat->get_size()[0]; ++i) {
        for (int j = 0; j < mat->get_size()[1]; ++j) {
            std::cout << mat->at(i, j) << " ";
        }
        std::cout << "\n";
    }
}
inline void print_mat(std::unique_ptr<dense_mtx>& mat)
{
    std::cout << "\n";
    for (int i = 0; i < mat->get_size()[0]; ++i) {
        for (int j = 0; j < mat->get_size()[1]; ++j) {
            std::cout << mat->at(i, j) << " ";
        }
        std::cout << "\n";
    }
}

inline void print_mat(mtx* mat, const std::shared_ptr<gko::OmpExecutor>& exec)
{
    auto dense_mat = gko::share(dense_mtx::create(exec));
    mat->convert_to(dense_mat.get());
    std::cout << "\n****************************************************\n";
    for (int i = 0; i < dense_mat->get_size()[0]; ++i) {
        for (int j = 0; j < dense_mat->get_size()[1]; ++j) {
            std::cout << dense_mat->at(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n****************************************************\n";
}