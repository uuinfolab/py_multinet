/*
 * Adapted from r_functions.h
 *
 * Created on: 2020-04-14
 * Author: matteomagnani
 * Version: 0.0.1
 */

#ifndef _PY_FUNCTIONS_H_
#define _PY_FUNCTIONS_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "PyMLNetwork.hpp"
#include "PyEvolutionModel.hpp"
#include <unordered_set>
#include <vector>
#include <memory>


namespace py = pybind11;


// CREATION AND STORAGE

PyMLNetwork
emptyMultilayer(
    const std::string& name
);



PyMLNetwork
readMultilayer(
               const std::string& input_file,
               const std::string& name,
               bool vertex_aligned
               );


void
writeMultilayer(
                const PyMLNetwork& mnet,
                const std::string& output_file,
                const std::string& format,
                const py::list& layer_names,
                char sep,
                bool merge_actors,
                bool all_actors
                );

// Evolution
 
PyEvolutionModel
ba_evolution_model(
    size_t m0,
    size_t m
);

PyEvolutionModel
er_evolution_model(
    size_t n
);

PyMLNetwork
growMultiplex(
    size_t num_actors,
    long num_of_steps,
    const py::list& evolution_model,
    const py::list& pr_internal_event,
    const py::list& pr_external_event,
    const py::list& dependency
);

py::dict
generateCommunities(
     const std::string& type,
     size_t num_actors,
     size_t num_layers,
     size_t num_communities,
     size_t overlap,
     const py::list& pr_internal,
     const py::list& pr_external
);

// INFORMATION ON NETWORKS

py::list
layers(
    const PyMLNetwork& mnet
);

py::dict
actors(
    const PyMLNetwork& mnet,
    const py::list& layer_names,
    bool add_attributes = false
);

py::dict
vertices(
    const PyMLNetwork& mnet,
    const py::list& layer_names,
    bool add_attributes = false
);

py::dict
edges(
    const PyMLNetwork& mnet,
    const py::list& layer_names1,
    const py::list& layer_names2,
    bool add_attributes = false
);

py::dict
edges_idx(
    const PyMLNetwork& rmnet
);

size_t
numLayers(
    const PyMLNetwork& mnet
);

size_t
numActors(
    const PyMLNetwork& mnet,
    const py::list& layers
);

size_t
numNodes(
    const PyMLNetwork& mnet,
    const py::list& layers
);

size_t
numEdges(
    const PyMLNetwork& mnet,
    const py::list& layer_names1,
    const py::list& layer_names2
);

py::dict
isDirected(
    const PyMLNetwork& mnet,
    const py::list& layer_names1,
    const py::list& layer_names2
);

std::unordered_set<std::string>
actor_neighbors(
    const PyMLNetwork& rmnet,
    const std::string& actor_name,
    const py::list& layer_names,
    const std::string& mode_name
);

std::unordered_set<std::string>
actor_xneighbors(
    const PyMLNetwork& rmnet,
    const std::string& actor_name,
    const py::list& layer_names,
    const std::string& mode_name
);


// NETWORK MANIPULATION

void
addLayers(
    PyMLNetwork& rmnet,
    const py::list& layer_names,
    const py::list& directed
);

void
addActors(
    PyMLNetwork& rmnet,
    const py::list& actor_names
);

void
addNodes(
    PyMLNetwork& rmnet,
    const py::dict& vertices
);

void
addEdges(
    PyMLNetwork& rmnet,
    const py::dict& edges);

void
setDirected(
    const PyMLNetwork&,
    const py::dict& directionalities
);

void
deleteLayers(
    PyMLNetwork& rmnet,
    const py::list& layer_names
);

void
deleteActors(
    PyMLNetwork& rmnet,
    const py::list& actor_names
);

void
deleteNodes(
    PyMLNetwork& rmnet,
    const py::dict& vertices
);

void
deleteEdges(
    PyMLNetwork& rmnet,
    const py::dict& edges
);



void
newAttributes(
    PyMLNetwork& rmnet,
    const py::list& attribute_names,
    const std::string& type,
    const std::string& target,
    const std::string& layer_name,
    const std::string& layer_name1,
    const std::string& layer_name2
);

py::dict
getAttributes(
    const PyMLNetwork&,
    const std::string& target
);

py::dict
getValues(
    const PyMLNetwork& rmnet,
    const std::string& attribute_name,
    const py::dict& actor_names,
    const py::dict& vertex_matrix,
    const py::dict& edge_matrix
);


void
setValues(
    PyMLNetwork& rmnet,
    const std::string& attribute_name,
    const py::dict& actor_names,
    const py::dict& vertex_matrix,
    const py::dict& edge_matrix,
    const py::list& values
);


// TRANSFORMATION

void
flatten(
    PyMLNetwork& rmnet,
    const std::string& new_layer,
    const py::list& layer_names,
    const std::string& method,
    bool force_directed,
    bool all_actors
);

void
project(
    PyMLNetwork& rmnet,
    const std::string& new_layer,
    const std::string& layer1,
    const std::string& layer2,
    const std::string& method
 );
 
// MEASURES

py::list
degree_ml(
    const PyMLNetwork&,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type
);


py::list
degree_deviation_ml(
    const PyMLNetwork&,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type
);

py::list
neighborhood_ml(
    const PyMLNetwork& mnet,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type
);

py::list
xneighborhood_ml(
    const PyMLNetwork& mnet,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type
);


py::list
connective_redundancy_ml(
    const PyMLNetwork& mnet,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type
);

py::list
relevance_ml(
    const PyMLNetwork& mnet,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type
);

py::list
xrelevance_ml(
    const PyMLNetwork& mnet,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type
);

double
summary_ml(
    const PyMLNetwork&,
    const std::string& layer,
    const std::string& method,
    const std::string& type
);

py::list
comparison_ml(
    const PyMLNetwork&,
    const py::list& layer_names,
    const std::string& method,
    const std::string& type,
    int K
);



py::dict
distance_ml(const PyMLNetwork& mnet,
            const std::string& from,
            const py::list& to,
            const std::string& method);


// CLUSTERING

py::dict
cliquepercolation_ml(
    const PyMLNetwork& rmnet,
    int k,
    int m
);


py::dict
infomap_ml(
           const PyMLNetwork& mnet,
           bool overlapping,
           bool directed,
           bool include_self_links
          );

py::dict
flat_ec(
    const PyMLNetwork& mnet
);

py::dict
flat_nw(
    const PyMLNetwork& mnet
);

py::dict
mdlp(
     const PyMLNetwork& mnet
);

py::dict
glouvain_ml(
    const PyMLNetwork&,
    double gamma,
    double omega
);

py::dict
abacus_ml(
    const PyMLNetwork&,
    int min_actors,
    int min_layers
);

double
nmi(
    const PyMLNetwork& rmnet,
    const py::dict& com1,
    const py::dict& com2
);

double
omega(
    const PyMLNetwork& rmnet,
    const py::dict& com1,
    const py::dict& com2
);


double
modularity_ml(
              const PyMLNetwork& rmnet,
              const py::dict& com, double gamma,
              double omega
              );

/*
List
to_list(
        const py::dict& cs,
        const PyMLNetwork& mnet
        );
*/

// Layout

py::dict
multiforce_ml(
    const PyMLNetwork& mnet,
    const py::list& w_in,
    const py::list& w_out,
    const py::list& gravity,
    int iterations
);

py::dict
circular_ml(
    const PyMLNetwork& mnet
);


/*

// SPREADING
// NumericMatrix sir(const PyMLNetwork& mnet, double beta, int tau, long num_iterations);

 */

py::dict
toNetworkxNodeDict(
                 const PyMLNetwork& rmnet
                 );

py::dict
toNetworkxEdgeDict(
               const PyMLNetwork& rmnet
               );

#endif
