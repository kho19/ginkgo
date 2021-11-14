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

// TODO: add description

#include <ginkgo/ginkgo.hpp>

#include <algorithm>
#include <cmath>
#include <random>

#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkDataSetAttributes.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPolyDataMapper.h>
#include <vtkProgrammableFilter.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

#include "helper.hpp"
#include "mesh.hpp"
using mtx = gko::matrix::Csr<>;
using dense_mtx = gko::matrix::Dense<>;
using executor = gko::ReferenceExecutor;

class timer : public vtkCommand {
public:
    timer(double tau, vtkProgrammableFilter* filter,
          vtkRenderWindowInteractor* interactor)
        : filter{filter}, interactor{interactor}
    {
        interactor->AddObserver(vtkCommand::TimerEvent, this);
        interactor->CreateRepeatingTimer(
            static_cast<unsigned long>(tau * 1000));
    }

    virtual void Execute(vtkObject* caller, unsigned long event_id, void*)
    {
        if (vtkCommand::TimerEvent == event_id) {
            filter->Modified();
            interactor->Render();
        }
    }

private:
    vtkProgrammableFilter* filter;
    vtkRenderWindowInteractor* interactor;
};


struct animation_state {
    vtkProgrammableFilter* filter;
    vtkDataArray* data;
    mesh* m;
    double time;
    double tau;
    double f;
    double k;
    std::unique_ptr<dense_mtx> u1;
    std::unique_ptr<dense_mtx> v1;
    std::unique_ptr<dense_mtx> u2;
    std::unique_ptr<dense_mtx> v2;
    std::unique_ptr<mtx> MminusA_u;
    std::unique_ptr<mtx> MminusA_v;
    std::unique_ptr<gko::solver::Cg<>> solver_u;
    std::unique_ptr<gko::solver::Cg<>> solver_v;
    std::shared_ptr<executor> exec;
};

// TODO: add comments explaining what is going on here
void generate_MA(const navigatable_mesh& m,
                 gko::matrix_assembly_data<double, int>& M,
                 gko::matrix_assembly_data<double, int>& A,
                 const std::shared_ptr<executor>& exec)
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

// Recursively set node values to 1 starting at seed edge and descending to
// specified depth
void init_uv_recursive(halfedge_id seed_edge, navigatable_mesh& m,
                       double* v_data, int levels = 3)
{
    --levels;
    auto next_edge = m.next_around_point(seed_edge);
    while (next_edge != seed_edge) {
        v_data[m.halfedges.at(next_edge).end] = 1.0;
        if (levels > 0) {
            auto next_seed_edge = m.halfedges.at(next_edge).opposite;
            init_uv_recursive(next_seed_edge, m, v_data, levels);
        }
        next_edge = m.next_around_point(next_edge);
    }
}

// Initialises the concentrations of the chemicals u and v
// u is initialised to 1 everwhere
// v is initialised to 0 everywhere apart from num_seed patches of depth
// seed_depth which are initialised to 1
void init_uv(dense_mtx* u, dense_mtx* v, navigatable_mesh& m, int num_seeds,
             int seed_depth)
{
    auto u_data = u->get_values();
    auto v_data = v->get_values();
    auto nelems = u->get_size()[0];
    // way through points
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, m.halfedges.size() / 2 - 1);
    std::vector<u_int> seed_edges;
    seed_edges.reserve(num_seeds);
    for (int i = 0; i < num_seeds; ++i) {
        seed_edges.emplace_back(dist(rng));
    }
    // populate initial conditions
    for (int i = 0; i < nelems; ++i) {
        // init u to 1.0 everywhere
        u_data[i] = 1.0;
        // init v to 0.0 everywhere
        v_data[i] = 0.0;
    }
    // iterate over seed edges setting nodes of branches to 1 recursively
    for (auto seed_edge : seed_edges) {
        v_data[m.halfedges.at(seed_edge).start] = 1.0;
        v_data[m.halfedges.at(seed_edge).end] = 1.0;
        init_uv_recursive(seed_edge, m, v_data, seed_depth);
    }
}

// explicit local update of the non-linear terms
void nonlin_update(dense_mtx* u, dense_mtx* v, double f, double k, double tau)
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

void animate(void* data)
{
    auto state = static_cast<animation_state*>(data);
    std::cout << "Global time: " << state->time << std::endl;
    state->time += state->tau;
    // Simulation step
    // Update diffusion term (half step: strang splitting)
    // TODO: better way to solve the systems? pack both into one large matrix
    state->MminusA_u->apply(gko::lend(state->u1), gko::lend(state->u2));
    state->solver_u->apply(gko::lend(state->u2), gko::lend(state->u1));
    state->MminusA_v->apply(gko::lend(state->v1), gko::lend(state->v2));
    state->solver_v->apply(gko::lend(state->v2), gko::lend(state->v1));

    // update nonlinear term
    // newton(state->u1, state->v1, state->f, state->k, state->tau,
    // state->exec);
    nonlin_update(gko::lend(state->u1), gko::lend(state->v1), state->f,
                  state->k, state->tau);
    // Update diffusion term
    state->MminusA_u->apply(gko::lend(state->u1), gko::lend(state->u2));
    state->solver_u->apply(gko::lend(state->u2), gko::lend(state->u1));
    state->MminusA_v->apply(gko::lend(state->v1), gko::lend(state->v2));
    state->solver_v->apply(gko::lend(state->v2), gko::lend(state->v1));

    for (int i = 0; i < state->m->points.size(); i++) {
        state->data->SetTuple1(i, state->v1->at(i, 0));
    }
    state->filter->GetPolyDataOutput()->CopyStructure(
        state->filter->GetPolyDataInput());
    auto point_data =
        state->filter->GetPolyDataOutput()->GetAttributes(vtkPolyData::POINT);
    point_data->SetScalars(state->data);
}


int main()
{
    /// Construct mesh from obj file
    vtkNew<vtkNamedColors> colors;
    // TODO: make input file interactive and add default independently from
    // build folder location
    std::ifstream stream{
        "../../../examples/fem-reaction-diffusion-equation/data/sphere6.obj"};
    auto init_m = parse_obj(stream);
    auto m = navigatable_mesh(init_m);
    auto poly_data = init_m.to_vtk();

    /// Define model parameters
    // diffusion factors
    auto Du = 0.0025;
    auto Dv = 0.00125;
    // feed and kill rates
    auto f = 0.038;
    auto k = 0.061;
    // number of simulation steps per second
    auto steps_per_sec = 4;
    // time step size for the simulation
    auto tau = 1.0 / steps_per_sec;

    // Set initialisation parameters
    int num_seeds = 2;
    int seed_depth = 4;

    /// Construct mass (M) and stiffness (A) matrices
    // TODO: consider adding interactive exec config like in poisson-solver
    auto exec = executor::create();
    auto nelems = m.points.size();
    auto M_data =
        gko::matrix_assembly_data<double, int>(gko::dim<2>(nelems, nelems));
    auto A_data =
        gko::matrix_assembly_data<double, int>(gko::dim<2>(nelems, nelems));

    generate_MA(m, M_data, A_data, exec);

    auto M = mtx::create(exec, gko::dim<2>(nelems, nelems));
    M->read(M_data);
    auto A = mtx::create(exec, gko::dim<2>(nelems, nelems));
    A->read(A_data);

    /// Construct matrices for explicit Euler
    // nxn unit matrix
    auto ones_data = gko::matrix_data<>::diag(gko::dim<2>(nelems, nelems), 1.0);
    auto ones = mtx::create(exec, nelems);
    ones->read(ones_data);
    auto beta = gko::initialize<dense_mtx>({1}, exec);

    // TODO: try tile these matrices into one big matrix

    // M + (tau/4)*Du*A
    auto alpha = gko::initialize<dense_mtx>({tau * Du / 4}, exec);
    auto MplusA_u = gko::clone(exec, M);
    A->apply(gko::lend(alpha), gko::lend(ones), gko::lend(beta),
             gko::lend(MplusA_u));

    // M - (tau/4)*Du*A
    alpha->at(0, 0) = -tau * Du / 4;
    auto MminusA_u = gko::clone(exec, M);
    A->apply(gko::lend(alpha), gko::lend(ones), gko::lend(beta),
             gko::lend(MminusA_u));

    // M + (tau/4)*Dv*A
    alpha->at(0, 0) = tau * Dv / 4;
    auto MplusA_v = gko::clone(exec, M);
    A->apply(gko::lend(alpha), gko::lend(ones), gko::lend(beta),
             gko::lend(MplusA_v));

    // M - (tau/4)*Dv*A
    alpha->at(0, 0) = -tau * Dv / 4;
    auto MminusA_v = gko::clone(exec, M);
    A->apply(gko::lend(alpha), gko::lend(ones), gko::lend(beta),
             gko::lend(MminusA_v));

    // combine system matrices for u and v into one large matrix
    // | MminusA_u     0   |
    // |    0    MminusA_v |
    auto LHS_data = gko::matrix_assembly_data<double, int>(
        gko::dim<2>(nelems * 2, nelems * 2));
    auto MA_u_data = MminusA_u->get_const_values();
    auto MA_u_row_ptr = MminusA_u->get_const_row_ptrs();
    auto MA_u_col_idxs = MminusA_u->get_const_col_idxs();
    auto MA_v_data = MminusA_v->get_const_values();
    auto MA_v_row_ptr = MminusA_v->get_const_row_ptrs();
    auto MA_v_col_idxs = MminusA_v->get_const_col_idxs();

    for (int i = 0; i < nelems; ++i) {
        for (int j = MA_u_row_ptr[i]; j < MA_u_row_ptr[i + 1]; ++j) {
            LHS_data.add_value(i, MA_u_col_idxs[j], MA_u_data[j]);
        }
        for (int j = MA_v_row_ptr[i]; j < MA_v_row_ptr[i + 1]; ++j) {
            LHS_data.add_value(i + nelems, MA_v_col_idxs[j] + nelems,
                               MA_v_data[j]);
        }
    }
    auto LHS = mtx::create(exec, gko::dim<2>(2 * nelems, 2 * nelems));
    LHS->read(LHS_data);

    // | MplusA_u     0   |
    // |    0    MplusA_v |
    auto RHS_data = gko::matrix_assembly_data<double, int>(
        gko::dim<2>(nelems * 2, nelems * 2));
    MA_u_data = MplusA_u->get_const_values();
    MA_u_row_ptr = MplusA_u->get_const_row_ptrs();
    MA_u_col_idxs = MplusA_u->get_const_col_idxs();
    MA_v_data = MplusA_v->get_const_values();
    MA_v_row_ptr = MplusA_v->get_const_row_ptrs();
    MA_v_col_idxs = MplusA_v->get_const_col_idxs();

    for (int i = 0; i < nelems; ++i) {
        for (int j = MA_u_row_ptr[i]; j < MA_u_row_ptr[i + 1]; ++j) {
            RHS_data.add_value(i, MA_u_col_idxs[j], MA_u_data[j]);
        }
        for (int j = MA_v_row_ptr[i]; j < MA_v_row_ptr[i + 1]; ++j) {
            RHS_data.add_value(i + nelems, MA_v_col_idxs[j] + nelems,
                               MA_v_data[j]);
        }
    }
    auto RHS = mtx::create(exec, gko::dim<2>(2 * nelems, 2 * nelems));
    RHS->read(RHS_data);

    /// Set initial conditions
    auto u1 = dense_mtx::create(exec, gko::dim<2>(nelems, 1));
    auto v1 = dense_mtx::create(exec, gko::dim<2>(nelems, 1));
    auto x1 = dense_mtx::create(exec, gko::dim<2>(2 * nelems, 1));
    auto x2 = dense_mtx::create(exec, gko::dim<2>(2 * nelems, 1));
    init_uv(gko::lend(u1), gko::lend(v1), m, num_seeds, seed_depth);

    for (int i = 0; i < nelems; ++i) {
        x1->at(i, 0) = u1->at(i, 0);
        x1->at(i + nelems, 0) = v1->at(i, 0);
    }

    /// Generate solver
    // use Ic or Jacobi
    auto solver_gen =
        gko::solver::Cg<>::build()
            .with_preconditioner(
                gko::preconditioner::Jacobi<>::build().on(exec))
            .with_criteria(gko::stop::RelativeResidualNorm<>::build()
                               .with_tolerance(1e-6)
                               .on(exec))
            .on(exec);

    auto solver_u = solver_gen->generate(gko::give(MplusA_u));
    auto solver_v = solver_gen->generate(gko::give(MplusA_v));

    vtkNew<vtkProgrammableFilter> filter;

    animation_state state{filter,
                          vtkDataArray::CreateDataArray(VTK_DOUBLE),
                          &init_m,
                          0.0,
                          tau,
                          f,
                          k,
                          std::move(u1),
                          std::move(v1),
                          std::move(u2),
                          std::move(v2),
                          std::move(MminusA_u),
                          std::move(MminusA_v),
                          std::move(solver_u),
                          std::move(solver_v),
                          std::move(exec)};
    state.data->SetNumberOfComponents(1);
    state.data->SetNumberOfTuples(init_m.points.size());
    filter->SetInputData(poly_data);
    filter->SetExecuteMethod(animate, &state);

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(filter->GetOutputPort());
    vtkNew<vtkActor> actor;
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(colors->GetColor3d("Silver").GetData());

    vtkNew<vtkRenderer> renderer;
    renderer->AddActor(actor);
    renderer->SetBackground(colors->GetColor3d("White").GetData());
    renderer->ResetCamera();
    renderer->GetActiveCamera()->Azimuth(30);
    renderer->GetActiveCamera()->Elevation(30);
    renderer->GetActiveCamera()->Dolly(1.5);
    renderer->ResetCameraClippingRange();

    vtkNew<vtkRenderWindow> renderWindow;
    renderWindow->AddRenderer(renderer);
    renderWindow->SetWindowName("objview");

    vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderWindowInteractor->Initialize();

    timer t{state.tau, filter, renderWindowInteractor};

    renderWindow->SetSize(1024, 768);
    renderWindow->Render();

    renderWindowInteractor->Start();
}