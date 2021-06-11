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
#include "community/Community.hpp"
#include "networks/MultilayerNetwork.hpp"

namespace py = pybind11;

std::vector<const uu::net::Network*>
resolve_const_layers(
    const uu::net::MultilayerNetwork* mnet,
    const py::list& names
);

std::vector<uu::net::Network*>
resolve_layers(
    uu::net::MultilayerNetwork* mnet,
    const py::list& names
);

std::unordered_set<uu::net::Network*>
resolve_layers_unordered(
    uu::net::MultilayerNetwork* mnet,
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

std::vector<std::pair<const uu::net::Vertex*, const uu::net::Network*>>
        resolve_const_vertices(
            const uu::net::MultilayerNetwork* mnet,
            const py::dict& vertex_matrix
        );

std::vector<std::tuple<const uu::net::Vertex*, const uu::net::Network*, const uu::net::Vertex*, const uu::net::Network*>>
        resolve_const_edges(
            const uu::net::MultilayerNetwork* mnet,
            const py::dict& edge_matrix
        );

std::vector<std::pair<const uu::net::Vertex*, uu::net::Network*>>
        resolve_vertices(
            uu::net::MultilayerNetwork* mnet,
            const py::dict& vertex_matrix
        );

std::vector<std::tuple<const uu::net::Vertex*, uu::net::Network*, const uu::net::Vertex*, uu::net::Network*>>
        resolve_edges(
            uu::net::MultilayerNetwork* mnet,
            const py::dict& edge_matrix
        );

uu::net::EdgeMode
resolve_mode(
    const std::string& mode
);

py::dict
to_dataframe(
    uu::net::CommunityStructure<uu::net::MultilayerNetwork>* cs
);

std::unique_ptr<uu::net::CommunityStructure<uu::net::MultilayerNetwork>>
to_communities(
               const py::dict& com,
               const uu::net::MultilayerNetwork* mnet
               );



#endif
