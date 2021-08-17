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

#pragma once

#include <array>
#include <ginkgo/ginkgo.hpp>
#include <istream>
#include <vector>

using point_id = int;
using halfedge_id = int;
using edge_id = int;
using triangle_id = int;

struct halfedge {
    halfedge(point_id start, point_id end)
        : start{start}, end{end}, opposite{-1}, edge{-1}
    {}

    point_id start;
    point_id end;
    halfedge_id opposite;
    edge_id edge;
};

struct mesh {
    std::vector<std::array<double, 3>> points;
    std::vector<std::array<point_id, 3>> triangles;
};

mesh parse_obj(std::istream &stream);
double tri_area(const std::array<point_id, 3> &tri, const mesh &m);


struct navigatable_mesh : mesh {
    navigatable_mesh(mesh m);
    std::vector<halfedge> halfedges;
    std::vector<halfedge_id> point_to_halfedge;
    std::vector<halfedge_id> edge_to_halfedge;

    triangle_id halfedge_to_triangle(halfedge_id e) const;
    halfedge_id triangle_to_halfedge(triangle_id e) const;
    halfedge_id next_around_triangle(halfedge_id e) const;
    halfedge_id next_around_point(halfedge_id e) const;
};

void tri_map_3D_2D(const std::array<point_id, 3> &tri, const mesh &m,
                   std::unique_ptr<gko::matrix::Dense<double>> &tri_2D,
                   double &area, const std::shared_ptr<gko::Executor> &exec);