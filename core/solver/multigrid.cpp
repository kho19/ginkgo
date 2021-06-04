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

#include <ginkgo/core/solver/multigrid.hpp>


#include <iostream>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/components/fill_array.hpp"
#include "core/solver/ir_kernels.hpp"
#include "core/solver/multigrid_kernels.hpp"


namespace gko {
namespace solver {
namespace multigrid {


GKO_REGISTER_OPERATION(initialize, ir::initialize);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(kcycle_step_1, multigrid::kcycle_step_1);
GKO_REGISTER_OPERATION(kcycle_step_2, multigrid::kcycle_step_2);
GKO_REGISTER_OPERATION(kcycle_check_stop, multigrid::kcycle_check_stop);


}  // namespace multigrid


namespace {


template <typename ValueType>
void handle_list(
    std::shared_ptr<const Executor> &exec, size_type index,
    std::shared_ptr<const LinOp> &matrix,
    std::vector<std::shared_ptr<const LinOpFactory>> &smoother_list,
    std::vector<std::shared_ptr<LinOp>> &smoother)
{
    auto list_size = smoother_list.size();
    if (list_size != 0) {
        auto temp_index = list_size == 1 ? 0 : index;
        GKO_ENSURE_IN_BOUNDS(temp_index, list_size);
        auto item = smoother_list.at(temp_index);
        if (item == nullptr) {
            smoother.emplace_back(nullptr);
        } else {
            auto solver = item->generate(matrix);
            if (solver->apply_uses_initial_guess() == true) {
                smoother.emplace_back(give(solver));
            } else {
                // if it is not use initial guess, use it as inner solver of Ir
                // with 1 iteration and relaxation_factor 1
                auto ir =
                    Ir<ValueType>::build()
                        .with_generated_solver(give(solver))
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters(1u).on(
                                exec))
                        .on(exec);
                smoother.emplace_back(give(ir->generate(matrix)));
            }
        }
    } else {
        smoother.emplace_back(nullptr);
    }
}


template <typename ValueType>
struct MultigridState {
    MultigridState(std::shared_ptr<const Executor> exec_in,
                   const LinOp *system_matrix_in,
                   const Multigrid<ValueType> *multigrid_in,
                   const size_type nrhs_in, const size_type k_base_in = 1,
                   const remove_complex<ValueType> rel_tol_in = 1)
        : exec{std::move(exec_in)},
          system_matrix(system_matrix_in),
          multigrid(multigrid_in),
          nrhs(nrhs_in),
          k_base(k_base_in),
          rel_tol(rel_tol_in)
    {
        auto current_nrows = system_matrix->get_size()[0];
        auto mg_level_list = multigrid->get_mg_level_list();
        auto list_size = mg_level_list.size();
        auto cycle = multigrid->get_cycle();
        r_list.reserve(list_size);
        g_list.reserve(list_size);
        e_list.reserve(list_size);
        one_list.reserve(list_size);
        neg_one_list.reserve(list_size);
        if (cycle == multigrid_cycle::kfcg || cycle == multigrid_cycle::kgcr) {
            auto k_num = mg_level_list.size() / k_base;
            alpha_list.reserve(k_num);
            beta_list.reserve(k_num);
            gamma_list.reserve(k_num);
            rho_list.reserve(k_num);
            zeta_list.reserve(k_num);
            v_list.reserve(k_num);
            w_list.reserve(k_num);
            d_list.reserve(k_num);
            old_norm_list.reserve(k_num);
            new_norm_list.reserve(k_num);
        }
        // Allocate memory first such that repeating allocation in each iter.
        for (int i = 0; i < mg_level_list.size(); i++) {
            auto next_nrows =
                mg_level_list.at(i)->get_coarse_op()->get_size()[0];
            auto mg_level = mg_level_list.at(i);
            // #define FOR_EACH(...)                                                       \
//     if (std::dynamic_pointer_cast<                                          \
//             const gko::multigrid::EnableMultigridLevel<HEAD(__VA_ARGS__)>>( \
//             mg_level)) {                                                    \
//         this->allocate_memory<HEAD(__VA_ARGS__)>(i, cycle, current_nrows,   \
//                                                  next_nrows);               \
//     } else {                                                                \
//         FOR_EACH(TAIL(...));                                                \
//     }

            //     FOR_EACH(float, double, complex<float>, complex<double>);

            if (std::dynamic_pointer_cast<
                    const gko::multigrid::EnableMultigridLevel<float>>(
                    mg_level)) {
                this->allocate_memory<float>(i, cycle, current_nrows,
                                             next_nrows);
            } else if (std::dynamic_pointer_cast<
                           const gko::multigrid::EnableMultigridLevel<double>>(
                           mg_level)) {
                this->allocate_memory<double>(i, cycle, current_nrows,
                                              next_nrows);
            } else if (std::dynamic_pointer_cast<
                           const gko::multigrid::EnableMultigridLevel<
                               std::complex<float>>>(mg_level)) {
                this->allocate_memory<std::complex<float>>(
                    i, cycle, current_nrows, next_nrows);
            } else if (std::dynamic_pointer_cast<
                           const gko::multigrid::EnableMultigridLevel<
                               std::complex<double>>>(mg_level)) {
                this->allocate_memory<std::complex<double>>(
                    i, cycle, current_nrows, next_nrows);
            } else {
                GKO_NOT_IMPLEMENTED;
            }

            current_nrows = next_nrows;
        }
    }

    template <typename VT>
    void allocate_memory(int level, multigrid_cycle cycle,
                         size_type current_nrows, size_type next_nrows)
    {
        using vec = matrix::Dense<VT>;
        using norm_vec = matrix::Dense<remove_complex<VT>>;

        r_list.emplace_back(vec::create(exec, dim<2>{current_nrows, nrhs}));
        g_list.emplace_back(vec::create(exec, dim<2>{next_nrows, nrhs}));
        e_list.emplace_back(vec::create(exec, dim<2>{next_nrows, nrhs}));
        std::static_pointer_cast<vec>(e_list.at(level))->fill(gko::zero<VT>());
        one_list.emplace_back(initialize<vec>({gko::one<VT>()}, exec));
        neg_one_list.emplace_back(initialize<vec>({-gko::one<VT>()}, exec));
        if ((cycle == multigrid_cycle::kfcg ||
             cycle == multigrid_cycle::kgcr) &&
            level % k_base == 0) {
            auto scaler_size = dim<2>{1, nrhs};
            auto vector_size = dim<2>{next_nrows, nrhs};
            // 1 x nrhs
            alpha_list.emplace_back(vec::create(exec, scaler_size));
            beta_list.emplace_back(vec::create(exec, scaler_size));
            gamma_list.emplace_back(vec::create(exec, scaler_size));
            rho_list.emplace_back(vec::create(exec, scaler_size));
            zeta_list.emplace_back(vec::create(exec, scaler_size));
            // next level's nrows x nrhs
            v_list.emplace_back(vec::create(exec, vector_size));
            w_list.emplace_back(vec::create(exec, vector_size));
            d_list.emplace_back(vec::create(exec, vector_size));
            // 1 x nrhs norm_vec
            old_norm_list.emplace_back(norm_vec::create(exec, scaler_size));
            new_norm_list.emplace_back(norm_vec::create(exec, scaler_size));
        }
    }

    void run_cycle(multigrid_cycle cycle, size_type level,
                   const std::shared_ptr<const LinOp> &matrix, const LinOp *b,
                   LinOp *x)
    {
        if (level == multigrid->get_mg_level_list().size()) {
            multigrid->get_coarsest_solver()->apply(b, x);
            return;
        }
        auto mg_level = multigrid->get_mg_level_list().at(level);
        if (std::dynamic_pointer_cast<
                const gko::multigrid::EnableMultigridLevel<float>>(mg_level)) {
            this->run_cycle<float>(cycle, level, matrix, b, x);
        } else if (std::dynamic_pointer_cast<
                       const gko::multigrid::EnableMultigridLevel<double>>(
                       mg_level)) {
            this->run_cycle<double>(cycle, level, matrix, b, x);
        } else if (std::dynamic_pointer_cast<
                       const gko::multigrid::EnableMultigridLevel<
                           std::complex<float>>>(mg_level)) {
            this->run_cycle<std::complex<float>>(cycle, level, matrix, b, x);
        } else if (std::dynamic_pointer_cast<
                       const gko::multigrid::EnableMultigridLevel<
                           std::complex<double>>>(mg_level)) {
            this->run_cycle<std::complex<double>>(cycle, level, matrix, b, x);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    }

    template <typename VT>
    void run_cycle(multigrid_cycle cycle, size_type level,
                   const std::shared_ptr<const LinOp> &matrix, const LinOp *b,
                   LinOp *x)
    {
        auto total_level = multigrid->get_mg_level_list().size();
#define as_vec(x) std::static_pointer_cast<matrix::Dense<VT>>(x)
#define as_real_vec(x) \
    std::static_pointer_cast<matrix::Dense<remove_complex<VT>>>(x)

        auto r = as_vec(r_list.at(level));
        auto g = as_vec(g_list.at(level));
        auto e = as_vec(e_list.at(level));
        // get mg_level
        auto mg_level = multigrid->get_mg_level_list().at(level);
        // get the pre_smoother
        auto pre_smoother = multigrid->get_pre_smoother_list().at(level);
        // get the mid_smoother
        auto mid_smoother = multigrid->get_mid_smoother_list().at(level);
        // get the post_smoother
        auto post_smoother = multigrid->get_post_smoother_list().at(level);
        auto one = one_list.at(level).get();
        auto neg_one = neg_one_list.at(level).get();
        // Smoother * x = r
        if (pre_smoother) {
            pre_smoother->apply(b, x);
            // compute residual
            r->copy_from(b);  // n * b
            matrix->apply(neg_one, x, one, r.get());
        } else if (level != 0) {
            // move the residual computation at level 0 to out-of-cycle if there
            // is no pre-smoother at level 0
            r->copy_from(b);
            matrix->apply(neg_one, x, one, r.get());
        }
        // first cycle
        mg_level->get_restrict_op()->apply(r.get(), g.get());
        // next level
        this->run_cycle(cycle, level + 1, mg_level->get_coarse_op(), g.get(),
                        e.get());
        // additional work for non-v_cycle
        if (cycle == multigrid_cycle::f || cycle == multigrid_cycle::w) {
            // second cycle - f_cycle, w_cycle
            // prolong
            mg_level->get_prolong_op()->apply(one, e.get(), one, x);
            // mid-smooth
            if (mid_smoother) {
                mid_smoother->apply(b, x);
            }
            // compute residual
            r->copy_from(b);  // n * b
            matrix->apply(neg_one, x, one, r.get());

            mg_level->get_restrict_op()->apply(r.get(), g.get());
            // next level
            if (cycle == multigrid_cycle::f) {
                // f_cycle call v_cycle in the second cycle
                run_cycle(multigrid_cycle::v, level + 1,
                          mg_level->get_coarse_op(), g.get(), e.get());
            } else {
                run_cycle(cycle, level + 1, mg_level->get_coarse_op(), g.get(),
                          e.get());
            }

        } else if ((cycle == multigrid_cycle::kfcg ||
                    cycle == multigrid_cycle::kgcr) &&
                   level % k_base == 0) {
            // otherwise, use v_cycle
            // do some work in coarse level - do not need prolong
            bool is_fcg = cycle == multigrid_cycle::kfcg;
            auto k_idx = level / k_base;
            auto alpha = as_vec(alpha_list.at(k_idx));
            auto beta = as_vec(beta_list.at(k_idx));
            auto gamma = as_vec(gamma_list.at(k_idx));
            auto rho = as_vec(rho_list.at(k_idx));
            auto zeta = as_vec(zeta_list.at(k_idx));
            auto v = as_vec(v_list.at(k_idx));
            auto w = as_vec(w_list.at(k_idx));
            auto d = as_vec(d_list.at(k_idx));
            auto old_norm = as_real_vec(old_norm_list.at(k_idx));
            auto new_norm = as_real_vec(new_norm_list.at(k_idx));
            auto matrix = mg_level->get_coarse_op();
            auto rel_tol_val = static_cast<remove_complex<VT>>(rel_tol);

            // first iteration
            matrix->apply(e.get(), v.get());
            std::shared_ptr<const matrix::Dense<VT>> t = is_fcg ? e : v;
            t->compute_dot(v.get(), rho.get());
            t->compute_dot(g.get(), alpha.get());

            if (!std::isnan(rel_tol_val) && rel_tol_val >= 0) {
                // calculate the r norm
                g->compute_norm2(old_norm.get());
            }
            // kcycle_step_1 update g, d
            // temp = alpha/rho
            // g = g - temp * v
            // d = e = temp * e
            exec->run(multigrid::make_kcycle_step_1(
                alpha.get(), rho.get(), v.get(), g.get(), d.get(), e.get()));
            // check ||new_r|| <= t * ||old_r|| only when t > 0 && t != inf
            bool is_stop = true;

            if (!std::isnan(rel_tol_val) && rel_tol_val >= 0) {
                // calculate the updated r norm
                g->compute_norm2(new_norm.get());
                // is_stop = true when all new_norm <= t * old_norm.
                exec->run(multigrid::make_kcycle_check_stop(
                    old_norm.get(), new_norm.get(), rel_tol_val, is_stop));
            }
            // rel_tol < 0: run two iteraion
            // rel_tol is nan: run one iteraions
            // others: new_norm <= rel_tol * old_norm -> run second iteraion.
            if (rel_tol_val < 0 || (rel_tol_val >= 0 && !is_stop)) {
                // second iteration
                // Apply on d for keeping the answer on e
                run_cycle(cycle, level + 1, mg_level->get_coarse_op(), g.get(),
                          d.get());
                matrix->apply(d.get(), w.get());
                t = is_fcg ? d : w;
                t->compute_dot(v.get(), gamma.get());
                t->compute_dot(w.get(), beta.get());
                t->compute_dot(g.get(), zeta.get());
                // kcycle_step_2 update e
                // scaler_d = zeta/(beta - gamma^2/rho)
                // scaler_e = 1 - gamma/alpha*scaler_d
                // e = scaler_e * e + scaler_d * d
                exec->run(multigrid::make_kcycle_step_2(
                    alpha.get(), rho.get(), gamma.get(), beta.get(), zeta.get(),
                    d.get(), e.get()));
            }
        }
        // prolong
        mg_level->get_prolong_op()->apply(one, e.get(), one, x);

        // post-smooth
        if (post_smoother) {
            post_smoother->apply(b, x);
        }
#undef as_vec
#undef as_real_vec
    }

    // current level's nrows x nrhs
    std::vector<std::shared_ptr<LinOp>> r_list;
    // next level's nrows x nrhs
    std::vector<std::shared_ptr<LinOp>> g_list;
    std::vector<std::shared_ptr<LinOp>> e_list;
    // Kcycle usage
    // 1 x nrhs
    std::vector<std::shared_ptr<LinOp>> alpha_list;
    std::vector<std::shared_ptr<LinOp>> beta_list;
    std::vector<std::shared_ptr<LinOp>> gamma_list;
    std::vector<std::shared_ptr<LinOp>> rho_list;
    std::vector<std::shared_ptr<LinOp>> zeta_list;
    std::vector<std::shared_ptr<LinOp>> old_norm_list;
    std::vector<std::shared_ptr<LinOp>> new_norm_list;
    // next level's nrows x nrhs
    std::vector<std::shared_ptr<LinOp>> v_list;
    std::vector<std::shared_ptr<LinOp>> w_list;
    std::vector<std::shared_ptr<LinOp>> d_list;
    // constant 1 x 1
    std::vector<std::shared_ptr<const LinOp>> one_list;
    std::vector<std::shared_ptr<const LinOp>> neg_one_list;
    std::shared_ptr<const Executor> exec;
    const LinOp *system_matrix;
    const Multigrid<ValueType> *multigrid;
    size_type nrhs;
    size_type k_base;
    remove_complex<ValueType> rel_tol;
};


}  // namespace


template <typename ValueType>
void Multigrid<ValueType>::generate()
{
    // generate coarse matrix until reaching max_level or min_coarse_rows
    auto num_rows = system_matrix_->get_size()[0];
    size_type level = 0;
    auto matrix = system_matrix_;
    auto exec = this->get_executor();
    std::cout << matrix->get_size()[0] << " " << matrix->get_size()[1]
              << std::endl;
    // Always generate smoother with size = level.
    while (level < parameters_.max_levels &&
           num_rows > parameters_.min_coarse_rows) {
        auto index = mg_level_index_(level, lend(matrix));
        GKO_ENSURE_IN_BOUNDS(index, parameters_.mg_level.size());
        auto mg_level_factory = parameters_.mg_level.at(index);
        // coarse generate
        std::cout << "123" << std::endl;
        auto mg_level = as<gko::multigrid::MultigridLevel>(
            share(mg_level_factory->generate(matrix)));
        if (mg_level->get_coarse_op()->get_size()[0] == num_rows) {
            // do not reduce dimension
            break;
        }
#define COPY                                                            \
    handle_list<VT>(exec, index, matrix, parameters_.pre_smoother,      \
                    pre_smoother_list_);                                \
    if (parameters_.mid_case == multigrid_mid_uses::mid) {              \
        handle_list<VT>(exec, index, matrix, parameters_.mid_smoother,  \
                        mid_smoother_list_);                            \
    }                                                                   \
    if (!parameters_.post_uses_pre) {                                   \
        handle_list<VT>(exec, index, matrix, parameters_.post_smoother, \
                        post_smoother_list_);                           \
    }

        std::cout << "generate " << level << std::endl;

        if (std::dynamic_pointer_cast<
                gko::multigrid::EnableMultigridLevel<float>>(mg_level)) {
            using VT = float;
            COPY
        } else if (std::dynamic_pointer_cast<
                       gko::multigrid::EnableMultigridLevel<double>>(
                       mg_level)) {
            using VT = double;
            COPY
        } else if (std::dynamic_pointer_cast<
                       gko::multigrid::EnableMultigridLevel<
                           std::complex<float>>>(mg_level)) {
            using VT = std::complex<float>;
            COPY
        } else if (std::dynamic_pointer_cast<
                       gko::multigrid::EnableMultigridLevel<
                           std::complex<double>>>(mg_level)) {
            using VT = std::complex<double>;
            COPY
        } else {
            GKO_NOT_IMPLEMENTED;
        }
        mg_level_list_.emplace_back(mg_level);
        matrix = mg_level_list_.back()->get_coarse_op();
        std::cout << num_rows << " -> " << matrix->get_size()[0] << std::endl;
        num_rows = matrix->get_size()[0];
        level++;
    }
    if (parameters_.post_uses_pre) {
        post_smoother_list_ = pre_smoother_list_;
    }
    if (parameters_.mid_case == multigrid_mid_uses::pre) {
        mid_smoother_list_ = pre_smoother_list_;
    } else if (parameters_.mid_case == multigrid_mid_uses::post) {
        mid_smoother_list_ = post_smoother_list_;
    }
    // Generate at least one level
    GKO_ASSERT_EQ(level > 0, true);
    auto last_mg_level = mg_level_list_.back();
#define SOLVER                                                                \
    if (parameters_.coarsest_solver.size() == 0) {                            \
        coarsest_solver_ =                                                    \
            matrix::Identity<VT>::create(exec, matrix->get_size()[0]);        \
    } else {                                                                  \
        auto temp_index = solver_index_(level, lend(matrix));                 \
        GKO_ENSURE_IN_BOUNDS(temp_index, parameters_.coarsest_solver.size()); \
        auto solver = parameters_.coarsest_solver.at(temp_index);             \
        if (solver == nullptr) {                                              \
            coarsest_solver_ =                                                \
                matrix::Identity<VT>::create(exec, matrix->get_size()[0]);    \
        } else {                                                              \
            coarsest_solver_ = solver->generate(matrix);                      \
        }                                                                     \
    }

    // generate coarsest solver
    if (std::dynamic_pointer_cast<
            const gko::multigrid::EnableMultigridLevel<float>>(last_mg_level)) {
        using VT = float;
        SOLVER
    } else if (std::dynamic_pointer_cast<
                   const gko::multigrid::EnableMultigridLevel<double>>(
                   last_mg_level)) {
        using VT = double;
        SOLVER
    } else if (std::dynamic_pointer_cast<
                   const gko::multigrid::EnableMultigridLevel<
                       std::complex<float>>>(last_mg_level)) {
        using VT = std::complex<float>;
        SOLVER
    } else if (std::dynamic_pointer_cast<
                   const gko::multigrid::EnableMultigridLevel<
                       std::complex<double>>>(last_mg_level)) {
        using VT = std::complex<double>;
        SOLVER
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType>
void Multigrid<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    auto exec = this->get_executor();
    constexpr uint8 RelativeStoppingId{1};
    Array<stopping_status> stop_status(exec, b->get_size()[1]);
    bool one_changed{};
    auto state = MultigridState<ValueType>(
        exec, system_matrix_.get(), this, b->get_size()[1],
        parameters_.kcycle_base, parameters_.kcycle_rel_tol);
    exec->run(multigrid::make_initialize(&stop_status));
    // compute the residual at the r_list(0);
    auto r = state.r_list.at(0);
    r->copy_from(b);
    system_matrix_->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, r.get());
    int iter = -1;
    while (true) {
        ++iter;
        this->template log<log::Logger::iteration_complete>(this, iter, r.get(),
                                                            x);
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .solution(x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        state.run_cycle(cycle_, 0, system_matrix_, b, x);
        r->copy_from(b);
        system_matrix_->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
    }
}


template <typename ValueType>
void Multigrid<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                      const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_MULTIGRID(_type) class Multigrid<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID);


}  // namespace solver
}  // namespace gko
