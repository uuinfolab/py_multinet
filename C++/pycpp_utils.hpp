/**
 *
 *
 * History:
 * - 2020.04.12 python version, adapted from the R library.
 * - 2018.09.12 updated to version 2.0 C++ uunet library API.
 * - 2014.07.29 file created.
 */

#ifndef UU_MULTINET_PYCPP_UTILS_H_
#define UU_MULTINET_PYCPP_UTILS_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_set>
#include <vector>
#include "community/CommunityStructure.hpp"
#include "objects/Vertex.hpp"
#include "objects/Edge.hpp"
#include "networks/Network.hpp"
#include "community/VertexLayerCommunity.hpp"
#include "networks/MultilayerNetwork.hpp"

namespace py = pybind11;

std::vector<uu::net::Network*>
resolve_layers(
    const uu::net::MultilayerNetwork* mnet,
    const py::list& names
);

std::unordered_set<uu::net::Network*>
resolve_layers_unordered(
    const uu::net::MultilayerNetwork* mnet,
    const py::list& names
);

std::unordered_set<const uu::net::Network*>
resolve_const_layers_unordered(
    const uu::net::MultilayerNetwork* mnet,
    const py::list& names
);

std::vector<const uu::net::Vertex*>
resolve_actors(
    const uu::net::MultilayerNetwork* mnet,
    const py::list& names
);

std::unordered_set<const uu::net::Vertex*>
resolve_actors_unordered(
    const uu::net::MultilayerNetwork* mnet,
    const py::list& names
);

std::vector<std::pair<const uu::net::Vertex*, uu::net::Network*>>
        resolve_vertices(
            const uu::net::MultilayerNetwork* mnet,
            const py::dict& vertex_matrix
        );

std::vector<std::tuple<const uu::net::Vertex*, uu::net::Network*, const uu::net::Vertex*, uu::net::Network*>>
        resolve_edges(
            const uu::net::MultilayerNetwork* mnet,
            const py::dict& edge_matrix
        );


uu::net::EdgeMode
resolve_mode(
    const std::string& mode
);

py::dict
to_dataframe(
    uu::net::CommunityStructure<uu::net::VertexLayerCommunity<const uu::net::Network>>* cs
);

std::unique_ptr<uu::net::CommunityStructure<uu::net::VertexLayerCommunity<const uu::net::Network>>>
to_communities(
               const py::dict& com,
               const uu::net::MultilayerNetwork* mnet
               );



#endif
