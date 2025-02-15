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

#include <ginkgo/core/base/index_set.hpp>


#include <random>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/index_set_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class IndexSet : public ::testing::Test {
protected:
    using index_type = T;
    IndexSet()
        : omp(gko::OmpExecutor::create()), ref(gko::ReferenceExecutor::create())
    {}

    gko::Array<index_type> setup_random_indices(int num_indices = 100)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<index_type> dist(0, num_indices);
        std::vector<index_type> index_vec(num_indices);
        for (auto& i : index_vec) {
            i = dist(gen);
        }
        auto rand_index_arr = gko::Array<index_type>(
            this->ref, index_vec.data(), num_indices + index_vec.data());
        return std::move(rand_index_arr);
    }

    static void assert_equal_index_sets(gko::IndexSet<T>& a,
                                        gko::IndexSet<T>& b)
    {
        ASSERT_EQ(a.get_size(), b.get_size());
        ASSERT_EQ(a.get_num_subsets(), b.get_num_subsets());
        if (a.get_num_subsets() > 0) {
            for (auto i = 0; i < a.get_num_subsets(); ++i) {
                EXPECT_EQ(a.get_subsets_begin()[i], b.get_subsets_begin()[i]);
                EXPECT_EQ(a.get_subsets_end()[i], b.get_subsets_end()[i]);
                EXPECT_EQ(a.get_superset_indices()[i],
                          b.get_superset_indices()[i]);
            }
        }
    }

    std::shared_ptr<const gko::OmpExecutor> omp;
    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(IndexSet, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(IndexSet, PopulateSubsetsIsEquivalentToReferenceForUnsortedInput)
{
    auto rand_arr = this->setup_random_indices(512);
    auto ref_begin_comp = gko::Array<TypeParam>{this->ref};
    auto ref_end_comp = gko::Array<TypeParam>{this->ref};
    auto ref_superset_comp = gko::Array<TypeParam>{this->ref};
    auto omp_begin_comp = gko::Array<TypeParam>{this->omp};
    auto omp_end_comp = gko::Array<TypeParam>{this->omp};
    auto omp_superset_comp = gko::Array<TypeParam>{this->omp};

    gko::kernels::reference::index_set::populate_subsets(
        this->ref, TypeParam(520), &rand_arr, &ref_begin_comp, &ref_end_comp,
        &ref_superset_comp, false);
    gko::kernels::omp::index_set::populate_subsets(
        this->omp, TypeParam(520), &rand_arr, &omp_begin_comp, &omp_end_comp,
        &omp_superset_comp, false);

    GKO_ASSERT_ARRAY_EQ(ref_begin_comp, omp_begin_comp);
    GKO_ASSERT_ARRAY_EQ(ref_end_comp, omp_end_comp);
    GKO_ASSERT_ARRAY_EQ(ref_superset_comp, omp_superset_comp);
}


TYPED_TEST(IndexSet, PopulateSubsetsIsEquivalentToReferenceForSortedInput)
{
    auto rand_arr = this->setup_random_indices(512);
    std::sort(rand_arr.get_data(),
              rand_arr.get_data() + rand_arr.get_num_elems());
    auto ref_begin_comp = gko::Array<TypeParam>{this->ref};
    auto ref_end_comp = gko::Array<TypeParam>{this->ref};
    auto ref_superset_comp = gko::Array<TypeParam>{this->ref};
    auto omp_begin_comp = gko::Array<TypeParam>{this->omp};
    auto omp_end_comp = gko::Array<TypeParam>{this->omp};
    auto omp_superset_comp = gko::Array<TypeParam>{this->omp};

    gko::kernels::reference::index_set::populate_subsets(
        this->ref, TypeParam(520), &rand_arr, &ref_begin_comp, &ref_end_comp,
        &ref_superset_comp, false);
    gko::kernels::omp::index_set::populate_subsets(
        this->omp, TypeParam(520), &rand_arr, &omp_begin_comp, &omp_end_comp,
        &omp_superset_comp, false);

    GKO_ASSERT_ARRAY_EQ(ref_begin_comp, omp_begin_comp);
    GKO_ASSERT_ARRAY_EQ(ref_end_comp, omp_end_comp);
    GKO_ASSERT_ARRAY_EQ(ref_superset_comp, omp_superset_comp);
}


TYPED_TEST(IndexSet, IndicesContainsIsEquivalentToReference)
{
    auto rand_arr = this->setup_random_indices(512);
    auto ref_idx_set = gko::IndexSet<TypeParam>(this->ref, 520, rand_arr);
    auto omp_idx_set = gko::IndexSet<TypeParam>(this->omp, 520, rand_arr);

    auto ref_indices_arr = this->setup_random_indices(73);
    auto ref_validity_arr = gko::Array<bool>(this->omp, 73);
    gko::kernels::reference::index_set::compute_validity(
        this->ref, &ref_indices_arr, &ref_validity_arr);
    auto omp_indices_arr = gko::Array<TypeParam>(this->omp, ref_indices_arr);
    auto omp_validity_arr = gko::Array<bool>(this->omp, 73);
    gko::kernels::omp::index_set::compute_validity(this->omp, &omp_indices_arr,
                                                   &omp_validity_arr);

    GKO_ASSERT_ARRAY_EQ(ref_validity_arr, omp_validity_arr);
}


TYPED_TEST(IndexSet, GetGlobalIndicesIsEquivalentToReference)
{
    auto rand_arr = this->setup_random_indices(512);
    auto rand_global_arr = this->setup_random_indices(256);
    auto ref_idx_set = gko::IndexSet<TypeParam>(this->ref, 520, rand_arr);
    auto ref_begin_comp = gko::Array<TypeParam>{
        this->ref, ref_idx_set.get_subsets_begin(),
        ref_idx_set.get_subsets_begin() + ref_idx_set.get_num_subsets()};
    auto ref_end_comp = gko::Array<TypeParam>{
        this->ref, ref_idx_set.get_subsets_end(),
        ref_idx_set.get_subsets_end() + ref_idx_set.get_num_subsets()};
    auto ref_superset_comp = gko::Array<TypeParam>{
        this->ref, ref_idx_set.get_superset_indices(),
        ref_idx_set.get_superset_indices() + ref_idx_set.get_num_subsets()};
    auto omp_idx_set = gko::IndexSet<TypeParam>(this->omp, 520, rand_arr);
    auto omp_begin_comp = gko::Array<TypeParam>{
        this->omp, omp_idx_set.get_subsets_begin(),
        omp_idx_set.get_subsets_begin() + omp_idx_set.get_num_subsets()};
    auto omp_end_comp = gko::Array<TypeParam>{
        this->omp, omp_idx_set.get_subsets_end(),
        omp_idx_set.get_subsets_end() + omp_idx_set.get_num_subsets()};
    auto omp_superset_comp = gko::Array<TypeParam>{
        this->omp, omp_idx_set.get_superset_indices(),
        omp_idx_set.get_superset_indices() + omp_idx_set.get_num_subsets()};

    auto ref_local_arr =
        gko::Array<TypeParam>{this->ref, rand_global_arr.get_num_elems()};
    gko::kernels::reference::index_set::global_to_local(
        this->ref, TypeParam(520), &ref_begin_comp, &ref_end_comp,
        &ref_superset_comp, &rand_global_arr, &ref_local_arr, false);
    auto omp_local_arr =
        gko::Array<TypeParam>{this->omp, rand_global_arr.get_num_elems()};
    gko::kernels::omp::index_set::global_to_local(
        this->omp, TypeParam(520), &omp_begin_comp, &omp_end_comp,
        &omp_superset_comp, &rand_global_arr, &omp_local_arr, false);

    ASSERT_EQ(rand_global_arr.get_num_elems(), omp_local_arr.get_num_elems());
    GKO_ASSERT_ARRAY_EQ(ref_local_arr, omp_local_arr);
}


TYPED_TEST(IndexSet, GetLocalIndicesIsEquivalentToReference)
{
    auto rand_arr = this->setup_random_indices(512);
    auto rand_local_arr = this->setup_random_indices(256);
    auto ref_idx_set = gko::IndexSet<TypeParam>(this->ref, 520, rand_arr);
    auto ref_begin_comp = gko::Array<TypeParam>{
        this->ref, ref_idx_set.get_subsets_begin(),
        ref_idx_set.get_subsets_begin() + ref_idx_set.get_num_subsets()};
    auto ref_end_comp = gko::Array<TypeParam>{
        this->ref, ref_idx_set.get_subsets_end(),
        ref_idx_set.get_subsets_end() + ref_idx_set.get_num_subsets()};
    auto ref_superset_comp = gko::Array<TypeParam>{
        this->ref, ref_idx_set.get_superset_indices(),
        ref_idx_set.get_superset_indices() + ref_idx_set.get_num_subsets()};
    auto omp_idx_set = gko::IndexSet<TypeParam>(this->omp, 520, rand_arr);
    auto omp_begin_comp = gko::Array<TypeParam>{
        this->omp, omp_idx_set.get_subsets_begin(),
        omp_idx_set.get_subsets_begin() + omp_idx_set.get_num_subsets()};
    auto omp_end_comp = gko::Array<TypeParam>{
        this->omp, omp_idx_set.get_subsets_end(),
        omp_idx_set.get_subsets_end() + omp_idx_set.get_num_subsets()};
    auto omp_superset_comp = gko::Array<TypeParam>{
        this->omp, omp_idx_set.get_superset_indices(),
        omp_idx_set.get_superset_indices() + omp_idx_set.get_num_subsets()};

    auto ref_global_arr =
        gko::Array<TypeParam>{this->ref, rand_local_arr.get_num_elems()};
    gko::kernels::reference::index_set::local_to_global(
        this->ref, TypeParam(520), &ref_begin_comp, &ref_end_comp,
        &ref_superset_comp, &rand_local_arr, &ref_global_arr, false);
    auto omp_global_arr =
        gko::Array<TypeParam>{this->omp, rand_local_arr.get_num_elems()};
    gko::kernels::omp::index_set::local_to_global(
        this->omp, TypeParam(520), &omp_begin_comp, &omp_end_comp,
        &omp_superset_comp, &rand_local_arr, &omp_global_arr, false);

    ASSERT_EQ(rand_local_arr.get_num_elems(), omp_global_arr.get_num_elems());
    GKO_ASSERT_ARRAY_EQ(ref_global_arr, omp_global_arr);
}


}  // namespace
