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

#include "mesh.hpp"

#include <ginkgo/ginkgo.hpp>
#include <limits>
#include <map>
#include <string>
#include "helper.hpp"


mesh parse_obj(std::istream& stream)
{
    mesh result;
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty()) {
            std::stringstream ss{line};
            std::string mode;
            ss >> mode;
            if (mode == "v") {
                double x{};
                double y{};
                double z{};
                ss >> x >> y >> z;
                result.points.emplace_back(std::array<double, 3>{x, y, z});
            } else if (mode == "f") {
                int a{};
                int b{};
                int c{};
                ss >> a >> b >> c;
                result.triangles.emplace_back(
                    std::array<int, 3>{a - 1, b - 1, c - 1});
            }
        }
    }
    return result;
}

double tri_area(const std::array<point_id, 3>& tri, const mesh& m)
{
    std::vector<double> temp1;
    temp1.reserve(3);
    std::vector<double> temp2;
    temp2.reserve(3);
    // Vector 1 -> 0
    temp1.at(0) = m.points.at(tri.at(0)).at(0) - m.points.at(tri.at(1)).at(0);
    temp1.at(1) = m.points.at(tri.at(0)).at(1) - m.points.at(tri.at(1)).at(1);
    temp1.at(2) = m.points.at(tri.at(0)).at(2) - m.points.at(tri.at(1)).at(2);
    // Vector 2 -> 0
    temp2.at(0) = m.points.at(tri.at(0)).at(0) - m.points.at(tri.at(2)).at(0);
    temp2.at(1) = m.points.at(tri.at(0)).at(1) - m.points.at(tri.at(2)).at(1);
    temp2.at(2) = m.points.at(tri.at(0)).at(2) - m.points.at(tri.at(2)).at(2);
    auto area =
        0.5 *
        std::sqrt(
            std::pow(temp1.at(1) * temp2.at(2) - temp1.at(2) * temp2.at(1),
                     (int)2) +
            std::pow(temp1.at(2) * temp2.at(0) - temp1.at(0) * temp2.at(2),
                     (int)2) +
            std::pow(temp1.at(0) * temp2.at(1) - temp1.at(1) * temp2.at(0),
                     (int)2));
    return area;
}

/*
 * Maps triangles in 3D space onto the xy plane using Givens rotations.
 * Absolute coordinates of the triangle vertices are obviously changed.
 * This does not affect entries in the stiffness matrix as these depend on the
 * products of gradients of linear functions which are not affected by
 * rotations. However, the association with a particular vertex (nodal basis
 * function) must be maintained for a correct end result.
 */
void tri_map_3D_2D(const std::array<point_id, 3>& tri, const mesh& m,
                   std::unique_ptr<gko::matrix::Dense<double>>& tri_2D,
                   double& area, const std::shared_ptr<gko::Executor>& exec)
{
    double nearly_zero = 1e-9;
    // two vectors that span the triangle
    std::vector<double> temp1;
    temp1.reserve(3);
    std::vector<double> temp2;
    temp2.reserve(3);
    // Givens rotations
    auto G = gko::initialize<gko::matrix::Dense<>>(
        {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, exec);
    auto G1 = gko::initialize<gko::matrix::Dense<>>(
        {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, exec);
    auto G2 = gko::initialize<gko::matrix::Dense<>>(
        {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, exec);
    // Givens parameters: hypotenuse, cosine, sine
    double r, c, s;

    // Vector 0 -> 1
    temp1.push_back(m.points.at(tri.at(1)).at(0) -
                    m.points.at(tri.at(0)).at(0));
    temp1.push_back(m.points.at(tri.at(1)).at(1) -
                    m.points.at(tri.at(0)).at(1));
    temp1.push_back(m.points.at(tri.at(1)).at(2) -
                    m.points.at(tri.at(0)).at(2));
    // Vector 0 -> 2
    temp2.push_back(m.points.at(tri.at(2)).at(0) -
                    m.points.at(tri.at(0)).at(0));
    temp2.push_back(m.points.at(tri.at(2)).at(1) -
                    m.points.at(tri.at(0)).at(1));
    temp2.push_back(m.points.at(tri.at(2)).at(2) -
                    m.points.at(tri.at(0)).at(2));

    // tri normal vector via cross product
    auto normal = gko::initialize<gko::matrix::Dense<>>(
        {temp1.at(1) * temp2.at(2) - temp1.at(2) * temp2.at(1),
         temp1.at(2) * temp2.at(0) - temp1.at(0) * temp2.at(2),
         temp1.at(0) * temp2.at(1) - temp1.at(1) * temp2.at(0)},
        exec);
    auto temp_normal = gko::matrix::Dense<>::create(exec, gko::dim<2>(3, 1));


    // Area of triangle using half the magnitude of the cross product vector
    area = 0.5 * std::sqrt(std::pow(normal->at(0), (int)2) +
                           std::pow(normal->at(1), (int)2) +
                           std::pow(normal->at(2), (int)2));
    assert(area > nearly_zero);

    // Givens rotation normal vector x comp -> 0
    // Signbit ensures that rotation always occurs to give a normal
    // vector pointing in z direction.
    auto sign =
        std::signbit(normal->at(0)) == std::signbit(normal->at(2)) ? 1.0 : -1.0;
    r = std::hypot(normal->at(0), normal->at(2));
    // If normal vector has only y component, no rotation needed in this stage
    if (r < nearly_zero) {
        c = 1;
        s = 0;
    } else {
        c = std::abs(normal->at(2)) / r;
        s = sign * std::abs(normal->at(0)) / r;
    }
    // Init and apply Givens matrix
    G1 = gko::initialize<gko::matrix::Dense<>>(
        {{c, 0.0, -s}, {0.0, 1.0, 0.0}, {s, 0.0, c}}, exec);
    G1->apply(gko::lend(normal), gko::lend(temp_normal));
    assert(temp_normal->at(0) < nearly_zero);

    // Givens rotation normal vector y comp -> 0
    sign = std::signbit(temp_normal->at(1)) == std::signbit(temp_normal->at(2))
               ? 1.0
               : -1.0;
    r = std::hypot(temp_normal->at(1), temp_normal->at(2));
    c = std::abs(temp_normal->at(2)) / r;
    s = sign * std::abs(temp_normal->at(1)) / r;
    G2 = gko::initialize<gko::matrix::Dense<>>(
        {{1.0, 0.0, 0.0}, {0.0, c, -s}, {0.0, s, c}}, exec);
    G2->apply(gko::lend(G1), gko::lend(G));

    /* reuse G1 for triangle verts
     * |x0, x1, x2|
     * |y0, y1, y2|
     * |z0, z1, z2|
     */
    int i = 0, j = 0;
    for (auto point_id : tri) {
        auto point = m.points.at(point_id);
        for (auto xyz : point) {
            // points as columns for givens rotation
            G1->at(j, i) = xyz;
            ++j;
        }
        j = 0;
        ++i;
    }

    // Test rotation on original normal vector
    G->apply(gko::lend(normal), gko::lend(temp_normal));
    assert(temp_normal->at(0) < nearly_zero);
    assert(temp_normal->at(1) < nearly_zero);
    auto len1 = std::pow(normal->at(0), (int)2) +
                std::pow(normal->at(1), (int)2) +
                std::pow(normal->at(2), (int)2);
    auto len2 = std::pow(temp_normal->at(0), (int)2) +
                std::pow(temp_normal->at(1), (int)2) +
                std::pow(temp_normal->at(2), (int)2);
    assert(len1 - len2 < nearly_zero);

    // apply givens rotation reusing G2
    G->apply(gko::lend(G1), gko::lend(G2));

    /* Construct stiffness matrix precursor tri_2D
     * |x2-x1, x0-x2, x1-x0|
     * |y2-y1, y0-y2, y1-y0|
     * Local contribution to stiffness matrix given by 3x3 matrix tri_2D'*tri_2D
     * / (4*tri_area) See https://en.wikipedia.org/wiki/Stiffness_matrix for
     * details
     */
    for (j = 0; j < 2; ++j) {
        for (i = 0; i < 3; ++i) {
            tri_2D->at(j, i) = G2->at(j, (2 + i) % 3) - G2->at(j, (1 + i) % 3);
        }
    }
}


vtkNew<vtkPolyData> mesh::to_vtk() const
{
    vtkNew<vtkPoints> out_points;
    out_points->SetNumberOfPoints(points.size());
    for (vtkIdType i = 0; i < points.size(); i++) {
        auto point = points[i];
        out_points->SetPoint(i, point[0], point[1], point[2]);
    }
    vtkNew<vtkCellArray> out_triangles;
    for (vtkIdType i = 0; i < triangles.size(); i++) {
        auto triangle = triangles[i];
        out_triangles->InsertNextCell(3);
        out_triangles->InsertCellPoint(triangle[0]);
        out_triangles->InsertCellPoint(triangle[1]);
        out_triangles->InsertCellPoint(triangle[2]);
    }
    vtkNew<vtkPolyData> out;
    out->SetPoints(out_points);
    out->SetPolys(out_triangles);
    return out;
}

navigatable_mesh::navigatable_mesh(mesh m) : mesh{std::move(m)}
{
    halfedges.reserve(triangles.size() * 3);
    edge_to_halfedge.reserve(triangles.size() * 3 / 2);
    // fill with max value of int
    point_to_halfedge.assign(points.size(),
                             std::numeric_limits<halfedge_id>::max());
    std::map<std::pair<point_id, point_id>, edge_id> edge_map;
    for (auto tri : triangles) {
        auto insert_halfedge = [&](point_id u, point_id v) {
            auto e_pair = std::make_pair(std::min(u, v), std::max(u, v));
            auto it = edge_map.find(e_pair);
            edge_id e_idx{-1};
            halfedge_id he_idx = halfedges.size();
            halfedges.emplace_back(u, v);
            auto& he = halfedges.back();
            // returned if edge was not found
            if (it == edge_map.end()) {
                // add a new edge, link edge -> halfedge
                e_idx = edge_to_halfedge.size();
                it = edge_map.emplace_hint(it, e_pair, e_idx);
                edge_to_halfedge.emplace_back(he_idx);
            } else {
                // use existing edge, link halfedge <-> halfedge
                e_idx = it->second;
                auto other_he_idx = edge_to_halfedge[e_idx];
                auto& other_he = halfedges[other_he_idx];
                he.opposite = other_he_idx;
                other_he.opposite = he_idx;
            }
            // link halfedge -> edge
            halfedges.back().edge = e_idx;
            // link points -> halfedge
            point_to_halfedge[u] = std::min(point_to_halfedge[u], he_idx);
            point_to_halfedge[v] = std::min(point_to_halfedge[v], he_idx);
        };
        insert_halfedge(tri[0], tri[1]);
        insert_halfedge(tri[1], tri[2]);
        insert_halfedge(tri[2], tri[0]);
    }
}

triangle_id navigatable_mesh::halfedge_to_triangle(halfedge_id e) const
{
    assert(e >= 0 && e < halfedges.size());
    return e / 3;
}

halfedge_id navigatable_mesh::triangle_to_halfedge(triangle_id t) const
{
    assert(t >= 0 && t < triangles.size());
    return t * 3;
}

halfedge_id navigatable_mesh::next_around_triangle(halfedge_id e) const
{
    assert(e >= 0 && e < halfedges.size());
    return (e / 3 * 3) + (e % 3 + 1) % 3;
}

halfedge_id navigatable_mesh::next_around_point(halfedge_id e) const
{
    assert(e >= 0 && e < halfedges.size());
    return halfedges[(e / 3 * 3) + (e % 3 + 2) % 3].opposite;
}