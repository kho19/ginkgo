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

#include <algorithm>
#include <fstream>
#include <random>
#include <string>

#include "mesh.hpp"

//#ifndef GKO_EXEC
//#define GKO_EXEC 1
//#endif

using mtx = gko::matrix::Csr<>;
using dense_mtx = gko::matrix::Dense<>;
#if GKO_EXEC == 1
using executor = gko::CudaExecutor;
#elif GKO_EXEC == 2
using executor = gko::OmpExecutor;
#elif GKO_EXEC == 3
using executor = gko::ReferenceExecutor;
#endif  // GKO_EXEC

void generate_MA(const navigatable_mesh &m,
                 gko::matrix_assembly_data<double, int> &M,
                 gko::matrix_assembly_data<double, int> &A,
                 const std::shared_ptr<executor> &exec)
{
    /* special helper matrix
     * |x2-x1, x0-x2, x1-x0|
     * |y2-y1, y0-y2, y1-y0|
     */
    auto tri_2D =
        gko::initialize<dense_mtx>({{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, exec);
    auto local_A = gko::initialize<dense_mtx>(
        {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, exec);
    auto alpha = gko::initialize<dense_mtx>({0.0}, exec);
    double tri_area = 0;

    for (auto tri : m.triangles) {
        // map triangle to 2D plane and calc area
        tri_map_3D_2D(tri, m, tri_2D, tri_area, exec);
        // calc local stiffness matrix A = tri_2D'*tri_2D/(4*tri_area)
        auto trans_tri_2D = tri_2D->transpose();
        trans_tri_2D->apply(gko::lend(tri_2D), gko::lend(local_A));
        alpha->at(0, 0) = 1 / (4 * tri_area);
        local_A->scale(gko::lend(alpha));
        auto delta_M = tri_area / 12;

        // Fill mass matrix (M) and stiffness matrix (A)
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                A.add_value(tri.at(i), tri.at(j), local_A->at(i, j));
                if (i == j) {
                    M.add_value(tri.at(i), tri.at(j), delta_M * 2);
                } else {
                    M.add_value(tri.at(i), tri.at(j), delta_M);
                }
            }
        }
    }
}

inline void set_init_val(const navigatable_mesh &m, const edge_id edge,
                         gko::matrix_data<> &init_data)
{
    for (auto &nonzero : init_data.nonzeros) {
        if (nonzero.row == m.halfedges.at(edge).end) {
            nonzero.value = 1.0;
            break;
        }
    }
}

inline void init_from_seed_edge(const halfedge_id seed_edge,
                                navigatable_mesh &m,
                                gko::matrix_data<> &init_v_data)
{
    halfedge_id level1_edge = m.next_around_point(seed_edge);
    // loop until starting point reached
    while (level1_edge != seed_edge) {
        // descend to required depth
        halfedge_id level2_seed_edge = m.halfedges.at(level1_edge).opposite;
        halfedge_id level2_edge = m.next_around_point(level2_seed_edge);

        while (level2_edge != level2_seed_edge) {
            set_init_val(m, level2_edge, init_v_data);
            level2_edge = m.next_around_point(level2_edge);
        }
        set_init_val(m, level1_edge, init_v_data);
        level1_edge = m.next_around_point(level1_edge);
    }
}

void init_uv(gko::matrix_data<> &init_u_data, gko::matrix_data<> &init_v_data,
             navigatable_mesh &m)
{
    int num_seeds = 2;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, m.halfedges.size() / 2 - 1);
    std::vector<int> seed_edges;
    seed_edges.reserve(num_seeds);
    for (int i = 0; i < num_seeds; ++i) {
        seed_edges.emplace_back(dist(rng));
    }
    // populate initial conditions
    for (int i = 0; i < init_u_data.size[0]; ++i) {
        // init u to 1.0 everywhere
        init_u_data.nonzeros.emplace_back(i, 0, 1.0);
        // init v to 0.0 everywhere
        init_v_data.nonzeros.emplace_back(i, 0, 0.0);
    }
    for (auto seed_edge : seed_edges) {
        init_from_seed_edge(seed_edge, m, init_v_data);
    }
}

void newton(std::shared_ptr<dense_mtx> &u, std::shared_ptr<dense_mtx> &v,
            double f, double k, double tau,
            const std::shared_ptr<executor> &exec)
{
    /*
     * Solve du/dt = F with F nonlinear u = [u,v] :P
     * u_n+1 = u_n + tau*F(u_n+1)
     * def G = u_n + tau*F(u_n+1) - u_n+1 and solve G = 0
     * Iterate u = u-delta_u
     * delta_u = (grad(G))^-1*G
     *
     *             |tau*dF2/dv-1  -tau*dF1/dv|
     * grad(G)^-1 =| |/((tau*dF1/du-1)*(tau*dF2/dv-1)-tau^2*dF2/du*dF1/dv)
     *             |-tau*dF2/du  tau*dF1/du-1|
     */
    auto nelems = u->get_num_stored_elements();
    auto alpha_one = gko::initialize<dense_mtx>({1}, exec);
    auto alpha_zero = gko::initialize<dense_mtx>({0}, exec);
    auto norm_du = gko::initialize<dense_mtx>({0}, exec);
    auto norm_dv = gko::initialize<dense_mtx>({0}, exec);

    // must work with row vectors here
    auto u0 = gko::as<dense_mtx>(u->transpose());
    auto v0 = gko::as<dense_mtx>(v->transpose());
    // init vectors for newton
    auto tauF1 = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto tauF2 = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto dG1du = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto dG1dv = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto dG2du = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto dG2dv = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto dets = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));


    // declare vectors that are updated every iteration
    auto G1 = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto G2 = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto u1 = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto v1 = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto du = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto dv = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));

    // initial guess for Newton
    u1->copy_from(gko::lend(u0));
    v1->copy_from(gko::lend(v0));
    int iter = 0;
    while (true) {
        // update F(x) and F'(x)
        // might be better to use ginkgo scale and add operations instead?
        for (int i = 0; i < nelems; ++i) {
            tauF1->at(0, i) =
                tau * (-u1->at(0, i) * (v1->at(0, i) * v1->at(0, i) + f) + f);
            tauF2->at(0, i) =
                tau * (v1->at(0, i) * (u1->at(0, i) * v1->at(0, i) - (f + k)));
            dG1du->at(0, i) = -tau * (v1->at(0, i) * v1->at(0, i) + f) - 1;
            dG1dv->at(0, i) = -tau * 2 * u1->at(0, i) * v1->at(0, i);
            dG2du->at(0, i) = tau * v1->at(0, i) * v1->at(0, i);
            dG2dv->at(0, i) =
                tau * (2 * u1->at(0, i) * v1->at(0, i) - (f + k)) - 1;
            dets->at(0, i) = (dG1du->at(0, i) * dG2dv->at(0, i) -
                              dG1dv->at(0, i) * dG2du->at(0, i));
        }
        G1->copy_from(gko::lend(u0));
        G1->sub_scaled(gko::lend(alpha_one), gko::lend(u1));
        G1->add_scaled(gko::lend(alpha_one), gko::lend(tauF1));

        G2->copy_from(gko::lend(v0));
        G2->sub_scaled(gko::lend(alpha_one), gko::lend(v1));
        G2->add_scaled(gko::lend(alpha_one), gko::lend(tauF2));

        du->copy_from(gko::lend(G1));
        du->scale(gko::lend(dG2dv));
        du->sub_scaled(gko::lend(dG2du), gko::lend(G2));
        du->inv_scale(gko::lend(dets));

        dv->copy_from(gko::lend(G2));
        dv->scale(gko::lend(dG1du));
        dv->sub_scaled(gko::lend(dG1dv), gko::lend(G1));
        dv->inv_scale(gko::lend(dets));

        // update u and v
        u1->sub_scaled(gko::lend(alpha_one), gko::lend(du));
        v1->sub_scaled(gko::lend(alpha_one), gko::lend(dv));
        // halt if accuracy requirement fulfilled
        gko::as<dense_mtx>(du->transpose())->compute_norm2(norm_du.get());
        gko::as<dense_mtx>(dv->transpose())->compute_norm2(norm_dv.get());

        //        std::cout << "Netwon iteration: " << iter;
        //        print_mat(norm_dv);
        ++iter;
#if GKO_EXEC == 1
        exec->synchronize();
#endif
        if (norm_du->at(0, 0) + norm_dv->at(0, 0) < 1e-6) break;
    }
    u->copy_from(gko::as<dense_mtx>(u1->transpose()));
    v->copy_from(gko::as<dense_mtx>(v1->transpose()));
}

void nonlin_update(std::shared_ptr<dense_mtx> &u, std::shared_ptr<dense_mtx> &v,
                   double f, double k, double tau)
{
    auto nelems = u->get_num_stored_elements();
    for (int i = 0; i < nelems; ++i) {
        u->at(i, 0) =
            u->at(i, 0) +
            tau * (-u->at(i, 0) * (v->at(i, 0) * v->at(i, 0) + f) + f);
        v->at(i, 0) =
            v->at(i, 0) +
            tau * (v->at(i, 0) * (u->at(i, 0) * v->at(i, 0) - (f + k)));
    }
}

int main()
{  /// Construct mesh from obj file
    std::ifstream stream{
        "/home/kit/stud/ubxin/ginkgo_personal_dev/examples/"
        "fem-reaction-diffusion-equation-plain-cuda/data/sphere6.obj"};
    auto init_m = parse_obj(stream);
    auto m = navigatable_mesh(init_m);

    /// Define parameters
    // these parameters produced sensible results for dragon
    auto Du = 0.00125;
    auto Dv = 0.000625;
    // feed and kill rates
    auto f = 0.038;
    auto k = 0.061;
    // number of simulation steps per second
    auto steps_per_sec = 4;
    // diffusion factors
    //    auto Du = 0.01;
    //    auto Dv = 0.005;
    //    // feed and kill rates
    //    auto f = 0.038;
    //    auto k = 0.061;
    //    // number of simulation steps per second
    //    auto steps_per_sec = 10;
    // time step size for the simulation
    auto tau = 1.0 / steps_per_sec;
    int t0 = tau * 101;
    /// Construct mass (M) and stiffness (A) matrices
    // actually use these to construct the matrices needed for implicit step

#if GKO_EXEC == 1
    auto exec = executor::create(0, gko::OmpExecutor::create(), true,
                                 gko::allocation_mode::unified_global);
#elif GKO_EXEC == 2 || GKO_EXEC == 3
    auto exec = executor::create();
#endif  // GKO_EXEC

    auto nelems = m.points.size();
    auto M_data =
        gko::matrix_assembly_data<double, int>(gko::dim<2>(nelems, nelems));
    auto A_data =
        gko::matrix_assembly_data<double, int>(gko::dim<2>(nelems, nelems));


    generate_MA(m, M_data, A_data, exec);

    std::cout << "M and A gen done\n";

    auto M = gko::share(mtx::create(exec));
    M->read(M_data);
    auto A = gko::share(mtx::create(exec));
    A->read(A_data);
    std::cout << "M and A init done\n";


    // write M and A to file
    //    std::ofstream
    //    Astream("../../../examples/fem-reaction-diffusion-equation/data/A.dat");
    //    std::ofstream
    //    Mstream("../../../examples/fem-reaction-diffusion-equation/data/M.dat");
    //    gko::write(Astream, gko::lend(A), gko::layout_type::coordinate);
    //    gko::write(Mstream, gko::lend(M), gko::layout_type::coordinate);
    //    return 0;


    /// Construct matrices for Crank-Nicolson
    // nxn unit matrix
    auto ones = gko::share(mtx::create(exec));
    auto ones_data = gko::matrix_data<>::diag(gko::dim<2>(nelems, nelems), 1.0);
    ones->read(ones_data);
    auto beta = gko::initialize<dense_mtx>({1}, exec);

    // M + (tau/2)*Du*A
    auto alpha = gko::initialize<dense_mtx>({tau * Du / 2}, exec);
    auto MplusA_u = gko::share(gko::clone(exec, M));
    A->apply(gko::lend(alpha), gko::lend(ones), gko::lend(beta),
             gko::lend(MplusA_u));

    // M - (tau/2)*Du*A
    alpha->at(0, 0) = -tau * Du / 2;
    auto MminusA_u = gko::share(gko::clone(exec, M));
    A->apply(gko::lend(alpha), gko::lend(ones), gko::lend(beta),
             gko::lend(MminusA_u));

    // M + (tau/2)*Dv*A
    alpha->at(0, 0) = tau * Dv / 2;
    auto MplusA_v = gko::share(gko::clone(exec, M));
    A->apply(gko::lend(alpha), gko::lend(ones), gko::lend(beta),
             gko::lend(MplusA_v));

    // M - (tau/2)*Dv*A
    alpha->at(0, 0) = -tau * Dv / 2;
    auto MminusA_v = gko::share(gko::clone(exec, M));
    A->apply(gko::lend(alpha), gko::lend(ones), gko::lend(beta),
             gko::lend(MminusA_v));

    std::cout << "CN matrix init done\n";

    /// Set initial conditions
    gko::matrix_data<> init_u_data{gko::dim<2>(m.points.size(), 1)};
    gko::matrix_data<> init_v_data{gko::dim<2>(m.points.size(), 1)};
    init_uv(init_u_data, init_v_data, m);
    auto u1 = gko::share(dense_mtx::create_with_type_of(
        alpha.get(), exec, gko::dim<2>(nelems, 1)));
    auto u2 = gko::share(dense_mtx::create_with_type_of(
        alpha.get(), exec, gko::dim<2>(nelems, 1)));
    auto v1 = gko::share(dense_mtx::create_with_type_of(
        alpha.get(), exec, gko::dim<2>(nelems, 1)));
    auto v2 = gko::share(dense_mtx::create_with_type_of(
        alpha.get(), exec, gko::dim<2>(nelems, 1)));
    u1->read(init_u_data);
    v1->read(init_v_data);
    std::cout << "Init conds init done\n";


    /// Generate solver
    // use Ic, Jacobi or GeneralIsai
    auto solver_gen =
        gko::solver::Cg<>::build()
            .with_preconditioner(
                gko::preconditioner::Jacobi<>::build().on(exec))
            .with_criteria(gko::stop::RelativeResidualNorm<>::build()
                               .with_tolerance(1e-6)
                               .on(exec))
            .on(exec);

    auto solver_u = solver_gen->generate(MplusA_u);
    auto solver_v = solver_gen->generate(MplusA_v);
    std::cout << "Solver init done\n";

    // CSV file for recording
    std::ofstream csv_out("./time_data.csv");
    csv_out << "cn,nonlin\n";


    /// Iterate
    for (double t = 0; t < t0; t += tau) {
        // Crank-Nicolson step for both equations
        // M*(u-u_old)/tau + (Du/4)*A*(u+u_old)
        auto start_cn = std::chrono::steady_clock::now();
        MminusA_u->apply(gko::lend(u1), gko::lend(u2));
        solver_u->apply(gko::lend(u2), gko::lend(u1));
        MminusA_v->apply(gko::lend(v1), gko::lend(v2));
        solver_v->apply(gko::lend(v2), gko::lend(v1));
        auto stop_cn = std::chrono::steady_clock::now();
        auto start_nonlin = std::chrono::steady_clock::now();

        // Newton or simple update for nonlinear term
        newton(u1, v1, f, k, tau, exec);
        //        nonlin_update(u1, v1, f, k, tau);

        auto stop_nonlin = std::chrono::steady_clock::now();
        auto runtime_cn =
            static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop_cn -
                                                                     start_cn)
                    .count()) *
            1e-6;
        auto runtime_nonlin =
            static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    stop_nonlin - start_nonlin)
                    .count()) *
            1e-6;
        std::cout << "Runtime cn (ms): " << runtime_cn
                  << "  runtime nonlin (ms): " << runtime_nonlin << "\n";
        csv_out << runtime_cn << "," << runtime_nonlin << "\n";
    }
    /// Clean up
    csv_out.close();
}