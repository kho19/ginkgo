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
#include <cmath>
#include <random>
#include <string>

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkDataSetAttributes.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>


#include "helper.hpp"
#include "mesh.hpp"

using mtx = gko::matrix::Csr<>;
using dense_mtx = gko::matrix::Dense<>;

void init_vtk_window(vtkNew<vtkPolyData> &poly_data,
                     vtkNew<vtkNamedColors> &colors,
                     vtkNew<vtkPolyDataMapper> &mapper, vtkNew<vtkActor> &actor,
                     vtkNew<vtkRenderer> &renderer,
                     vtkNew<vtkRenderWindow> &renderWindow, navigatable_mesh &m)
{
    vtkColor3d back_color = colors->GetColor3d("White");
    vtkColor3d model_color = colors->GetColor3d("Silver");

    auto point_data = poly_data->GetAttributes(vtkPolyData::POINT);
    auto scalar_data = vtkDataArray::CreateDataArray(VTK_DOUBLE);
    for (int i = 0; i < m.points.size(); i++) {
        scalar_data->InsertTuple1(i, std::sin(m.points[i][0]) +
                                         std::cos(m.points[i][1]) +
                                         std::sin(m.points[i][2]));
    }
    point_data->SetScalars(scalar_data);

    mapper->SetInputData(poly_data);
    actor->SetMapper(mapper);
    actor->GetProperty()->SetDiffuseColor(model_color.GetData());

    renderer->AddActor(actor);
    renderer->SetBackground(back_color.GetData());
    renderer->ResetCamera();
    renderer->GetActiveCamera()->Azimuth(30);
    renderer->GetActiveCamera()->Elevation(30);
    renderer->GetActiveCamera()->Dolly(1.5);
    renderer->ResetCameraClippingRange();

    renderWindow->AddRenderer(renderer);
    renderWindow->SetWindowName("objview");

    vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderWindow->SetSize(1024, 768);
    renderWindow->Render();

    renderWindowInteractor->Start();
}


void generate_MA(const navigatable_mesh &m,
                 gko::matrix_assembly_data<double, int> &M,
                 gko::matrix_assembly_data<double, int> &A,
                 const std::shared_ptr<gko::OmpExecutor> &exec)
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

void init_uv(gko::matrix_data<> &init_u_data, gko::matrix_data<> &init_v_data)
{
    // choose three random seeds for v
    int clump_size = 5;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, init_u_data.size[0] - (clump_size + 1));
    std::vector<int> dots{0, 0, 0};
    for (auto &dot : dots) {
        dot = dist(rng);
    }
    std::sort(dots.begin(), dots.end(), std::greater<>());
    // populate initial conditions
    auto dot = dots.back();
    dots.pop_back();

    for (int i = 0; i < init_u_data.size[0]; ++i) {
        // init u to 1.0 everywhere
        init_u_data.nonzeros.emplace_back(i, 0, 1.0);
        if (i >= 30 && i < 30 + clump_size)
            init_v_data.nonzeros.emplace_back(i, 0, 1.0);
        else
            init_v_data.nonzeros.emplace_back(i, 0, 0.0);
        // init v to 0.0 with random clumps of 1.0
        //        if (i >= dot && i < dot + clump_size) {
        //            init_v_data.nonzeros.emplace_back(i, 0, 1.0);
        //        } else if (i == dot + clump_size) {
        //            init_v_data.nonzeros.emplace_back(i, 0, 0.0);
        //            dot = dots.back();
        //            dots.pop_back();
        //        } else {
        //            init_v_data.nonzeros.emplace_back(i, 0, 0.0);
        //        }
    }

    /// Alternate initialisation
    //    // choose random seed triangles for v
    //    int num_seeds = 10;
    //    std::random_device dev;
    //    std::mt19937 rng(dev());
    //    std::uniform_int_distribution<std::mt19937::result_type> dist(
    //        0, m.triangles.size()-1);
    //    std::vector<u_int> seed_tris;
    //    seed_tris.reserve(num_seeds);
    //    for (int i = 0; i < num_seeds; ++i) {
    //        seed_tris.emplace_back(dist(rng));
    //    }
    //    // populate initial conditions
    //    for (int i = 0; i < init_u_data.size[0]; ++i) {
    //        // init u to 1.0 everywhere
    //        init_u_data.nonzeros.emplace_back(i, 0, 1.0);
    //        // init v to 0.0 everywhere
    //        init_v_data.nonzeros.emplace_back(i, 0, 0.0);
    //    }
    //    for (auto tri_idx:seed_tris){
    //        auto tri = m.triangles.at(tri_idx);
    //        for (auto point:tri) {
    //            for (auto & nonzero : init_v_data.nonzeros) {
    //                if (nonzero.row == point) {
    //                    nonzero.value = 1.0;
    //                    break;
    //                }
    //            }
    //        }
    //    }
}

void newton(std::shared_ptr<dense_mtx> &u, std::shared_ptr<dense_mtx> &v,
            double f, double k, double tau,
            const std::shared_ptr<gko::OmpExecutor> &exec)
{
    /*
     * Solve du/dt = F with F nonlinear u = [u,v] :P
     * u_n+1 = u_n + tau*F(u_n)
     * def G = u_n + tau*F(u_n) - u_n+1 and solve G = 0
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

    for (int i = 0; i < nelems; ++i) {
        tauF1->at(0, i) =
            tau * (-u0->at(0, i) * (v0->at(0, i) * v0->at(0, i) + f) + f);
        tauF2->at(0, i) =
            tau * (v0->at(0, i) * (u0->at(0, i) * v0->at(0, i) - (f + k)));
        dG1du->at(0, i) = -tau * (v0->at(0, i) * v0->at(0, i) + f) - 1;
        dG1dv->at(0, i) = -tau * 2 * u0->at(0, i) * v0->at(0, i);
        dG2du->at(0, i) = tau * v0->at(0, i) * v0->at(0, i);
        dG2dv->at(0, i) = tau * (2 * u0->at(0, i) * v0->at(0, i) - (f + k)) - 1;
        dets->at(0, i) = (dG1du->at(0, i) * dG2dv->at(0, i) -
                          dG1dv->at(0, i) * dG2du->at(0, i));
    }

    // declare vectors that are updated every iteration
    auto G1 = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto G2 = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto u2 = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto v2 = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto du = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));
    auto dv = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(1, nelems)));

    u2->copy_from(gko::lend(u0));
    v2->copy_from(gko::lend(v0));
    while (true) {
        G1->copy_from(gko::lend(u0));
        G1->sub_scaled(gko::lend(alpha_one), gko::lend(u2));
        G1->add_scaled(gko::lend(alpha_one), gko::lend(tauF1));

        G2->copy_from(v0.get());
        G2->sub_scaled(gko::lend(alpha_one), gko::lend(v2));
        G2->add_scaled(gko::lend(alpha_one), gko::lend(tauF2));

        du->copy_from(gko::lend(G1));
        du->scale(gko::lend(dG2dv));
        du->sub_scaled(gko::lend(dG1dv), gko::lend(G2));
        du->inv_scale(gko::lend(dets));

        dv->copy_from(gko::lend(G2));
        dv->scale(gko::lend(dG1du));
        dv->sub_scaled(gko::lend(dG2du), gko::lend(G1));
        dv->inv_scale(gko::lend(dets));

        // update u and v
        u2->sub_scaled(gko::lend(alpha_one), gko::lend(du));
        v2->sub_scaled(gko::lend(alpha_one), gko::lend(dv));
        // halt if accuracy requirement fulfilled
        gko::as<dense_mtx>(du->transpose())->compute_norm2(norm_du.get());
        gko::as<dense_mtx>(dv->transpose())->compute_norm2(norm_dv.get());

        if (norm_du->at(0, 0) + norm_dv->at(0, 0) < 1e-3) break;
    }
    u->copy_from(gko::as<dense_mtx>(u2->transpose()));
    v->copy_from(gko::as<dense_mtx>(v2->transpose()));
}

int main()
{
    /// Define parameters
    // simulation length
    auto t0 = 50.000;
    // diffusion factors
    auto Du = 1.0;
    auto Dv = 0.5;
    // feed and kill rates
    auto f = 0.055;
    auto k = 0.062;
    // number of simulation steps per second
    auto steps_per_sec = 4;
    // number of video frames per second
    auto fps = 25;
    // time step size for the simulation
    auto tau = 1.0 / steps_per_sec;


    /// Construct mesh from obj file
    std::ifstream stream{
        "../../../examples/fem-reaction-diffusion-equation/data/sphere.obj"};
    auto init_m = parse_obj(stream);
    auto m = navigatable_mesh(init_m);

    /// Render mesh with vtk
    //        auto poly_data = m.to_vtk();
    //        vtkNew<vtkNamedColors> colors;
    //        vtkNew<vtkPolyDataMapper> mapper;
    //        vtkNew<vtkActor> actor;
    //        vtkNew<vtkRenderer> renderer;
    //        vtkNew<vtkRenderWindow> renderWindow;
    //        init_vtk_window(poly_data, colors, mapper, actor, renderer,
    //        renderWindow, m);

    /// Construct mass (M) and stiffness (A) matrices
    // actually use these to construct the matrices needed for implicit step
    auto exec = gko::ReferenceExecutor::create();
    //        auto exec = gko::OmpExecutor::create();
    auto nelems = m.points.size();
    auto M_data =
        gko::matrix_assembly_data<double, int>(gko::dim<2>(nelems, nelems));
    auto A_data =
        gko::matrix_assembly_data<double, int>(gko::dim<2>(nelems, nelems));

    generate_MA(m, M_data, A_data, exec);

    auto M = gko::share(mtx::create(exec));
    M->read(M_data);
    auto A = gko::share(mtx::create(exec));
    A->read(A_data);

    /// Construct matrices for Crank-Nicolson
    // nxn unit matrix
    auto ones = gko::share(mtx::create(exec));
    auto ones_data = gko::matrix_data<>::diag(gko::dim<2>(nelems, nelems), 1.0);
    ones->read(ones_data);
    auto beta = gko::initialize<dense_mtx>({1}, exec);

    // M + (tau/4)*Du*A
    auto alpha = gko::initialize<dense_mtx>({tau * Du / 4}, exec);
    auto MplusA_u = gko::share(gko::clone(exec, M));
    A->apply(gko::lend(alpha), gko::lend(ones), gko::lend(beta),
             gko::lend(MplusA_u));

    // M - (tau/4)*Du*A
    alpha->at(0, 0) = -tau * Du / 4;
    auto MminusA_u = gko::share(gko::clone(exec, M));
    A->apply(gko::lend(alpha), gko::lend(ones), gko::lend(beta),
             gko::lend(MminusA_u));

    // M + (tau/4)*Dv*A
    alpha->at(0, 0) = tau * Dv / 4;
    auto MplusA_v = gko::share(gko::clone(exec, M));
    A->apply(gko::lend(alpha), gko::lend(ones), gko::lend(beta),
             gko::lend(MplusA_v));

    // M - (tau/4)*Dv*A
    alpha->at(0, 0) = -tau * Dv / 4;
    auto MminusA_v = gko::share(gko::clone(exec, M));
    A->apply(gko::lend(alpha), gko::lend(ones), gko::lend(beta),
             gko::lend(MminusA_v));

    /// Set initial conditions
    gko::matrix_data<> init_u_data{gko::dim<2>(m.points.size(), 1)};
    gko::matrix_data<> init_v_data{gko::dim<2>(m.points.size(), 1)};
    init_uv(init_u_data, init_v_data);
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

    /// Generate solvers
    auto solver_gen =
        gko::solver::Cg<>::build()
            .with_preconditioner(gko::preconditioner::Ic<>::build().on(exec))
            .with_criteria(gko::stop::RelativeResidualNorm<>::build()
                               .with_tolerance(1e-10)
                               .on(exec))
            .on(exec);

    auto solver_u = solver_gen->generate(MplusA_u);
    auto solver_v = solver_gen->generate(MplusA_v);

    /// Iterate
    for (double t = 0; t < t0; t += tau) {
        // Half Crank-Nicolson step for both equations
        // M*(u-u_old)/tau + (Du/4)*A*(u+u_old)
        auto start_time = std::chrono::steady_clock::now();
        MminusA_u->apply(gko::lend(u1), gko::lend(u2));
        solver_u->apply(gko::lend(u2), gko::lend(u1));
        MminusA_v->apply(gko::lend(v1), gko::lend(v2));
        solver_v->apply(gko::lend(v2), gko::lend(v1));

        // Newton for nonlinear term
        newton(u1, v1, f, k, tau, exec);


        // Remaining Crank-Nicolson half step
        MminusA_u->apply(gko::lend(u1), gko::lend(u2));
        solver_u->apply(gko::lend(u2), gko::lend(u1));
        MminusA_v->apply(gko::lend(v1), gko::lend(v2));
        solver_v->apply(gko::lend(v2), gko::lend(v1));
        auto stop_time = std::chrono::steady_clock::now();
        auto runtime = static_cast<double>(
                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                               stop_time - start_time)
                               .count()) *
                       1e-6;
        std::cout << "Runtime (ms): " << runtime << "\n";
    }
    /// Clean up
}