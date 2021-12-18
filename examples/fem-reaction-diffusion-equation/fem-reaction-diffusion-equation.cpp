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

/*****************************<DESCRIPTION>***********************************
This example solves a 2D reaction diffusion equation

\begin{equation}
\begin{aligned}
\partial_tu &= \delta_u\Delta u-uv^2+f(1-u),\\
    \partial_tv &= \delta_v\Delta v+uv^2-(f+k)v.
    \end{aligned}
\end{equation}

using the finite element method on a 2D manifold without boundaries, with given
initial conditions and fixed parameters $\delta_u, \delta_v, f, k$.

The varying concentration of two reacting and diffusing chemicals $U$ and $V$
are taken to represent the pigment concentration on a two dimensional manifold.
By changing the parameters $\delta_u$, $\delta_v$, $f$ and $k$ different
patterns form.

The parameters $\delta_u$ and $\delta_v$ represent the diffusion coefficients of
each chemical respectively. The term $uv^2$ accounts for the reaction $U +
2V\rightarrow 3V$ that converts $U$ into $V$. The chemical $U$ is replenished at
a rate proportional to $1-u$ given by the feed rate $f$. Finally, the term
$-(f+k)v$ models the reaction $V\rightarrow P$ to an inert product $P$
counteracting the buildup of $V$.

Equation \eqref{eq:teq} is a time-dependent semi-linear elliptic PDE and as such
cannot be directly solved using finite elements. Strang splitting is used to
handle the non-linearity. This involves numerical treatment of the diffusion
term and the reaction term in separate steps. Equation \eqref{eq:teq} is split
as shown in equation \eqref{eq:split} and the approximate solutions of $u$ and
$v$ in \eqref{eq:split1} and \eqref{eq:split2} are updated alternately.
\begin{subequations}
\label{eq:split}
\begin{equation}
\label{eq:split1}
\begin{aligned}
\partial_tu &= \delta_u\Delta u,\\
\partial_tv &= \delta_v\Delta v.
\end{aligned}
\end{equation}
\begin{equation}
\label{eq:split2}
\begin{aligned}
\partial_tu &=-uv^2+f(1-u),\\
\partial_tv &=uv^2-(f+k)v.
\end{aligned}
\end{equation}
\end{subequations}

The diffusion term in equation \eqref{eq:split1} is approximated using the FEM.
This results in a system of ODE which is approximated using the Crank-Nicolson
method with a step size of $\tau$ resulting in the linear system
\begin{equation}
\label{eq:cn}
(M+\delta_u\frac{\tau}{2}A)\cdot \bm{x}_n = (M-\delta_u\frac{\tau}{2}A)\cdot
\bm{x}_{n-1}
\end{equation}
which must be solved in each step.

The non-linear reaction term \eqref{eq:split2} is approximated using an
explicit Euler update. All the nodal function values are directly updated in one
step according to
\begin{align}
x^i_n &= x^i_{n-1} + \tau (-x^i_{n-1}(y^i_{n-1})^2 + f(1-x^i_{n-1})),\\
y^i_n &= y^i_{n-1} + \tau (x^i_{n-1}(y^i_{n-1})^2 - (f+k)y^i_{n-1}) \quad
i\in\{1,\dots,n_i\}. \end{align}. Similar to the heat equation example, the
intention of this example is to provide a mini-app showing matrix assembly,
 vector initialization, solver setup. This example is more complicated than
 the heat equation.
*****************************<DESCRIPTION>**********************************/

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

#include <ginkgo/kernels/kernel_declaration.hpp>

#include "helper.hpp"
#include "mesh.hpp"
using mtx = gko::matrix::Csr<>;
using dense_mtx = gko::matrix::Dense<>;

GKO_DECLARE_UNIFIED(void nonlinear_update(
    std::shared_ptr<const DefaultExecutor> exec, int nelems, double tau,
    double f, double k, dense_mtx* x));

GKO_REGISTER_UNIFIED_OPERATION(nonlinear_update, nonlinear_update);

// Default config values
// diffusion factors
constexpr double def_Du = 0.0025;
constexpr double def_Dv = 0.00125;
// feed and kill rates
constexpr double def_f = 0.038;
constexpr double def_k = 0.061;
// number of simulation steps per second
constexpr double def_steps_per_sec = 4;
// initialisation parameters
int def_num_seeds = 3;
int def_seed_depth = 3;

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
    std::unique_ptr<dense_mtx> x1;
    std::unique_ptr<dense_mtx> x2;
    std::unique_ptr<dense_mtx> cpu_x1;
    std::unique_ptr<mtx> RHS;
    std::unique_ptr<gko::solver::Cg<>> solver_x;
    std::shared_ptr<gko::Executor> exec;
};

/* Mass (M) and stiffness (A) matrices are generated.
 * To use well known methods for 2D each triangle is oriented to lie in the xy
 * plane using Givens rotations and z coordinate is ignored. The error
 * introduced by this simplification is related to the curvature of the 3D mesh
 * at that point and the area spanned by a nodal basis function. Using a finer
 * mesh i.e. decreasing the area of the support of the nodal basis function
 * should decrease the error of this simplification.
 */
void generate_MA(const navigatable_mesh& m,
                 gko::matrix_assembly_data<double, int>& M,
                 gko::matrix_assembly_data<double, int>& A,
                 const std::shared_ptr<gko::Executor>& exec)
{
    /* Construct stiffness matrix precursor tri_2D
     * |x2-x1, x0-x2, x1-x0|
     * |y2-y1, y0-y2, y1-y0|
     * Local contribution to stiffness matrix given by 3x3 matrix tri_2D'*tri_2D
     * / (4*tri_area) See https://en.wikipedia.org/wiki/Stiffness_matrix for
     * details
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

        // local mass matrix M fully determined by the triangles area.
        // off diagonals = tri_area/12. diagonals = tri_area/6.
        // Results from 2nd order quadrature rule on triangles with nodes at
        // edge midpoints
        auto delta_M = tri_area / 12;

        // Fill mass matrix (M) and stiffness matrix (A)
        // Entry A(i,j) corresponds with interaction between nodal basis
        // functions from nodes i and j
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
                       double* v_data, int levels)
{
    --levels;
    if (levels > 0) {
        auto next_edge = m.next_around_point(seed_edge);
        while (next_edge != seed_edge) {
            auto next_seed_edge = m.halfedges.at(next_edge).opposite;
            init_uv_recursive(next_seed_edge, m, v_data, levels);
            v_data[m.halfedges.at(next_edge).end] = 1.0;
            next_edge = m.next_around_point(next_edge);
        }
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
    // generate random list of seed edges
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

void animate(void* data)
{
    auto state = static_cast<animation_state*>(data);
    auto nelems = state->x1->get_num_stored_elements() / 2;

    // perform 10 steps for each update of the graphic
    for (int i = 0; i < 10; ++i) {
        std::cout << "Global time: " << state->time << std::endl;
        state->time += state->tau;
        // Simulation step
        // Update diffusion term (half step: strang splitting)
        state->RHS->apply(gko::lend(state->x1), gko::lend(state->x2));
        state->solver_x->apply(gko::lend(state->x2), gko::lend(state->x1));

        // update nonlinear term
        state->exec->run(make_nonlinear_update(nelems, state->tau, state->f,
                                               state->k, gko::lend(state->x1)));
        // Update diffusion term
        state->RHS->apply(gko::lend(state->x1), gko::lend(state->x2));
        state->solver_x->apply(gko::lend(state->x2), gko::lend(state->x1));
    }

    // update graphic
    // copy solution back to cpu for visualisation
    state->cpu_x1->copy_from(state->x1.get());
    for (int i = 0; i < state->m->points.size(); i++) {
        state->data->SetTuple1(i, state->cpu_x1->at(nelems + i, 0));
    }
    state->filter->GetPolyDataOutput()->CopyStructure(
        state->filter->GetPolyDataInput());
    auto point_data =
        state->filter->GetPolyDataOutput()->GetAttributes(vtkPolyData::POINT);
    point_data->SetScalars(state->data);
}


int main(int argc, char** argv)
{
    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && std::string(argv[1]) == "--help" ||
        std::string(argv[1]) == "-h") {
        std::cerr << "Usage: " << argv[0]
                  << " [mesh] [executor] [Du] [Dv] [f] [k] [steps/sec] [num "
                     "seeds] [seed depth]"
                  << std::endl;
        std::exit(-1);
    }

    auto mesh_string_temp = argc >= 2 ? argv[1] : "sphere";
    const auto exec_string = argc >= 3 ? argv[2] : "reference";
    // diffusion factors
    const auto Du = argc >= 4 ? std::strtod(argv[3], (char**)nullptr) : def_Du;
    const auto Dv = argc >= 5 ? std::strtod(argv[4], (char**)nullptr) : def_Dv;
    // feed and kill rates
    const auto f = argc >= 6 ? std::strtod(argv[5], (char**)nullptr) : def_f;
    const auto k = argc >= 7 ? std::strtod(argv[6], (char**)nullptr) : def_k;
    // number of simulation steps per second
    const auto steps_per_sec =
        argc >= 8 ? std::strtod(argv[7], (char**)nullptr) : def_steps_per_sec;
    // Set initialisation parameters
    const auto num_seeds =
        argc >= 9 ? std::strtol(argv[8], (char**)nullptr, 10) : def_num_seeds;
    const auto seed_depth =
        argc >= 10 ? std::strtol(argv[9], (char**)nullptr, 10) : def_seed_depth;
    // time step size for the simulation
    const auto tau = 1.0 / steps_per_sec;
    // TODO: Do not know if this will run on gpu
    //  Figure out where to run the code
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                 true);
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(exec_string)();  // throws if not valid
    // executor where the application initialized the data
    const auto cpu_exec = exec->get_master();

    // mesh obj file to be used
    std::stringstream ss;
    ss << "data/" << mesh_string_temp << ".obj";
    const auto mesh_string = ss.str();
    std::ifstream stream{mesh_string};

    // Construct mesh from obj file
    vtkNew<vtkNamedColors> colors;
    auto init_m = parse_obj(stream);
    auto m = navigatable_mesh(init_m);
    auto poly_data = init_m.to_vtk();

    // Construct mass (M) and stiffness (A) matrices
    // Matrices and vectors are constructed on the GPU (if present)
    // Solution vector is copied back to the CPU at regular intervals to update
    // the visualisation
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

    // Construct matrices for explicit Euler
    // nxn unit matrix
    auto ones_data = gko::matrix_data<>::diag(gko::dim<2>(nelems, nelems), 1.0);
    auto ones = mtx::create(exec, nelems);
    ones->read(ones_data);
    auto beta = gko::initialize<dense_mtx>({1}, exec);

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
    auto RHS_data = gko::matrix_assembly_data<double, int>(
        gko::dim<2>(nelems * 2, nelems * 2));
    auto MA_u_data = MminusA_u->get_values();
    auto MA_u_row_ptr = MminusA_u->get_const_row_ptrs();
    auto MA_u_col_idxs = MminusA_u->get_const_col_idxs();
    auto MA_v_data = MminusA_v->get_values();
    auto MA_v_row_ptr = MminusA_v->get_const_row_ptrs();
    auto MA_v_col_idxs = MminusA_v->get_const_col_idxs();

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

    // | MplusA_u     0   |
    // |    0    MplusA_v |
    auto LHS_data = gko::matrix_assembly_data<double, int>(
        gko::dim<2>(nelems * 2, nelems * 2));
    MA_u_data = MplusA_u->get_values();
    MA_u_row_ptr = MplusA_u->get_const_row_ptrs();
    MA_u_col_idxs = MplusA_u->get_const_col_idxs();
    MA_v_data = MplusA_v->get_values();
    MA_v_row_ptr = MplusA_v->get_const_row_ptrs();
    MA_v_col_idxs = MplusA_v->get_const_col_idxs();

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

    // Set initial conditions
    auto u1 = dense_mtx::create(exec, gko::dim<2>(nelems, 1));
    auto v1 = dense_mtx::create(exec, gko::dim<2>(nelems, 1));
    auto x1 = dense_mtx::create(exec, gko::dim<2>(2 * nelems, 1));
    auto x2 = dense_mtx::create(exec, gko::dim<2>(2 * nelems, 1));
    init_uv(gko::lend(u1), gko::lend(v1), m, num_seeds, seed_depth);

    for (int i = 0; i < nelems; ++i) {
        x1->at(i, 0) = u1->at(i, 0);
        x1->at(i + nelems, 0) = v1->at(i, 0);
    }

    auto cpu_x1 = gko::clone(cpu_exec, x1);

    // Generate solver
    // use Ic or Jacobi
    auto solver_gen =
        gko::solver::Cg<>::build()
            .with_preconditioner(
                gko::preconditioner::Jacobi<>::build().on(exec))
            .with_criteria(gko::stop::RelativeResidualNorm<>::build()
                               .with_tolerance(1e-6)
                               .on(exec))
            .on(exec);
    auto solver_x = solver_gen->generate(gko::give(LHS));

    vtkNew<vtkProgrammableFilter> filter;

    animation_state state{filter,
                          vtkDataArray::CreateDataArray(VTK_DOUBLE),
                          &init_m,
                          0.0,
                          tau,
                          f,
                          k,
                          std::move(x1),
                          std::move(x2),
                          std::move(cpu_x1),
                          std::move(RHS),
                          std::move(solver_x),
                          exec};
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