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

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>

#include "mesh.hpp"

using mtx = gko::matrix::Csr<>;
using dense_mtx = gko::matrix::Dense<>;

void init_vtk_window(vtkNew<vtkPolyData> &poly_data,
                     vtkNew<vtkNamedColors> &colors,
                     vtkNew<vtkPolyDataMapper> &mapper, vtkNew<vtkActor> &actor,
                     vtkNew<vtkRenderer> &renderer,
                     vtkNew<vtkRenderWindow> &renderWindow)
{
    vtkColor3d back_color = colors->GetColor3d("White");
    vtkColor3d model_color = colors->GetColor3d("Silver");
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
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, init_u_data.size[0] - 11);
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
        // init v to 0.0 with random clumps of 1.0
        if (i >= dot && i < dot + 10) {
            init_v_data.nonzeros.emplace_back(i, 0, 1.0);
        } else if (i == dot + 10) {
            init_v_data.nonzeros.emplace_back(i, 0, 0.0);
            dot = dots.back();
            dots.pop_back();
        } else {
            init_v_data.nonzeros.emplace_back(i, 0, 0.0);
        }
    }
}

void newton(std::shared_ptr<dense_mtx> &u_in, std::shared_ptr<dense_mtx> &u_out,
            double &F, double &k, const std::shared_ptr<gko::OmpExecutor> &exec)
{
    /*
     * Solve du/dt = F with F nonlinear u = [u,v] :P
     * u_n+1 = u_n + tau*F(u_n)
     * def G = u_n + tau*F(u_n) - u_n+1 and solve G = 0
     * Iterate u = u-delta_u
     * delta_u = (grad(G))^-1*G
     *
     *             |tau*dF2/dv - 1  -tau*dF1/dv|
     * grad(G)^-1 =| |/((tau*dF1/du-1)*(tau*dF2/dv-1)-tau^2*dF2/du*dF1/dv)
     *             |-tau*dF2/du  tau*dF1/du - 1|
     */
    auto nelems = u_in->get_num_stored_elements();
    auto alpha_one = gko::initialize<dense_mtx>({1}, exec);
    auto alpha_zero = gko::initialize<dense_mtx>({0}, exec);
    // TODO: no better way to copy values from one matrix to another?

    auto test = alpha_zero->get_const_values();
    u_out->scale(gko::lend(alpha_zero));
    u_out->add_scaled(gko::lend(alpha_one), gko::lend(u_in));


    // init vectors for newton
    auto G1 = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(nelems, 1)));
    auto G2 = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(nelems, 1)));
    auto dG1du = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(nelems, 1)));
    auto dG1dv = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(nelems, 1)));
    auto dG2du = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(nelems, 1)));
    auto dG2dv = gko::share(dense_mtx::create_with_type_of(
        alpha_one.get(), exec, gko::dim<2>(nelems, 1)));

    // evaluate G
}

int main()
{
    /// Define parameters
    // simulation length
    auto t0 = 5.0;
    // diffusion factors
    auto Du = 0.0005;
    auto Dv = 0.0005;
    // feed and kill rates
    auto f = 0.055;
    auto k = 0.062;
    // number of simulation steps per second
    auto steps_per_sec = 500;
    // number of video frames per second
    auto fps = 25;
    // time step size for the simulation
    auto tau = 1.0 / steps_per_sec;


    /// Construct mesh from obj file
    // TODO: mesh construction is not complete, edges and association functions
    // missing
    std::ifstream stream{
        "../../../examples/fem-reaction-diffusion-equation/data/dragon.obj"};
    auto init_m = parse_obj(stream);
    auto m = navigatable_mesh(init_m);

    /// Render mesh with vtk
    // TODO: Rendering must include colour of the node values and be updated
    // each iteration
    //    auto poly_data = m.to_vtk();
    //    vtkNew<vtkNamedColors> colors;
    //    vtkNew<vtkPolyDataMapper> mapper;
    //    vtkNew<vtkActor> actor;
    //    vtkNew<vtkRenderer> renderer;
    //    vtkNew<vtkRenderWindow> renderWindow;
    //    init_vtk_window(poly_data, colors, mapper, actor, renderer,
    //    renderWindow);

    /// Construct mass (M) and stiffness (A) matrices
    // actually use these to construct the matrices needed for implicit step
    auto exec = gko::OmpExecutor::create();
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
    auto beta = gko::initialize<dense_mtx>({0}, exec);

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
    // TODO: need way of storing nodal values on mesh for display
    gko::matrix_data<> init_u_data{gko::dim<2>(m.points.size(), 1)};
    gko::matrix_data<> init_v_data{gko::dim<2>(m.points.size(), 1)};
    init_uv(init_u_data, init_v_data);
    auto u_in = gko::share(dense_mtx::create_with_type_of(
        alpha.get(), exec, gko::dim<2>(nelems, 1)));
    auto u_out = gko::share(dense_mtx::create_with_type_of(
        alpha.get(), exec, gko::dim<2>(nelems, 1)));
    auto v_in = gko::share(dense_mtx::create_with_type_of(
        alpha.get(), exec, gko::dim<2>(nelems, 1)));
    auto v_out = gko::share(dense_mtx::create_with_type_of(
        alpha.get(), exec, gko::dim<2>(nelems, 1)));
    u_out->read(init_u_data);
    v_out->read(init_v_data);

    /// Generate solvers
    auto solver_gen =
        gko::solver::Cg<>::build()
            .with_preconditioner(gko::preconditioner::Ic<>::build().on(exec))
            .with_criteria(gko::stop::RelativeResidualNorm<>::build()
                               .with_tolerance(1e-5)
                               .on(exec))
            .on(exec);

    auto solver_u = solver_gen->generate(MplusA_u);
    auto solver_v = solver_gen->generate(MplusA_v);

    /// Iterate
    for (double t = 0; t < t0; t += tau) {
        // Half Crank-Nicolson step for both equations
        // M*(u-u_old)/tau + (Du/4)*A*(u+u_old)
        MminusA_u->apply(gko::lend(u_out), gko::lend(u_in));
        solver_u->apply(gko::lend(u_in), gko::lend(u_out));
        std::swap(u_in, u_out);
        MminusA_v->apply(gko::lend(v_out), gko::lend(v_in));
        solver_v->apply(gko::lend(v_in), gko::lend(v_out));
        std::swap(v_in, v_out);

        // Newton for nonlinear term
        newton(u_in, u_out, f, k, exec);

        // Remaining Crank-Nicolson half step
        MminusA_u->apply(gko::lend(u_out), gko::lend(u_in));
        solver_u->apply(gko::lend(u_in), gko::lend(u_out));
        std::swap(u_in, u_out);
        MminusA_v->apply(gko::lend(v_out), gko::lend(v_in));
        solver_v->apply(gko::lend(v_in), gko::lend(v_out));
        std::swap(v_in, v_out);

        // Visualise results: write to frame and/or video
    }
    /// Clean up
}