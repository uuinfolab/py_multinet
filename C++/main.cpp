#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "py_functions.hpp"

namespace py = pybind11;

using namespace py::literals;

PYBIND11_MODULE(_multinet, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: _multinet

        .. autosummary::
           :toctree: _generate
    
    )pbdoc";

    py::class_<PyMLNetwork>(m, "PyMLNetwork")
    .def(py::init<std::shared_ptr<uu::net::AttributedHomogeneousMultilayerNetwork>>())
    .def("__repr__",
         [](const PyMLNetwork& net) {
             return net.get_mlnet()->summary();
         }
         );
    
    py::class_<PyEvolutionModel>(m, "PyEvolutionModel")
    .def(py::init<std::shared_ptr<uu::net::EvolutionModel<uu::net::AttributedHomogeneousMultilayerNetwork>>,
         std::string>())
    .def("__repr__",
         [](const PyEvolutionModel& mod) {
             return mod.description();
         }
         );
    
    m.def("empty", &emptyMultilayer,
          py::arg("name") = "", R"pbdoc(
        )pbdoc");

    m.def("read", &readMultilayer,
          py::arg("file"),
          py::arg("name") = "unnamed",
          py::arg("sep") = ',',
          py::arg("aligned") = false,
    R"pbdoc(Reads a multilayer network from a file)pbdoc");

    m.def("write", &writeMultilayer,
          py::arg("n"),
          py::arg("file"),
          py::arg("format") = "multilayer",
          py::arg("layers") = py::list(),
          py::arg("sep") = ',',
          py::arg("merge.actors") = true,
          py::arg("all.actors") = false,
    R"pbdoc(Writes a multilayer network to a file)pbdoc");
    
    m.def("evolution_pa", &ba_evolution_model,
          py::arg("m0"),
          py::arg("m"),
    R"pbdoc(Creates a layer evolutionary model based on preferential attachment)pbdoc");
    
    m.def("evolution_er", &er_evolution_model,
          py::arg("n"),
    R"pbdoc(Creates a layer evolutionary model based on random edge creation, as in the ER model)pbdoc");
    
    m.def("grow_ml",&growMultiplex,
          py::arg("num.actors"),
          py::arg("num.steps"),
          py::arg("models"),
          py::arg("pr.internal"),
          py::arg("pr.external"),
          py::arg("dependency"),
    R"pbdoc(Grows a multiplex network)pbdoc");
    
    /**************************************/
    /* INFORMATION ON MULTILAYER NETWORKS */
    /**************************************/
    
    m.def("layers",
             &layers,
             py::arg("n"),
    R"pbdoc(Returns the list of layers in the input multilayer network)pbdoc");
    
    m.def("actors", &actors,
        py::arg("n"),
        py::arg("layers") = py::list(),
    R"pbdoc(Returns the list of actors present in the input layers, or in the whole multilayer network if no layers are specified)pbdoc");
    
    m.def("vertices", &vertices,
             py::arg("n"),
          py::arg("layers") = py::list(),
    R"pbdoc(Returns the list of vertices in the input layers, or in the whole multilayer network if no layers are specified)pbdoc");
    
    m.def("edges", &edges,
             py::arg("n"),
          py::arg("layers1") = py::list(),
          py::arg("layers2") = py::list(),
    R"pbdoc(Returns the list of edges among vertices in the input layers (if only one set of layers is specified), or from the first set of input layers to the second set of input layers, or in the whole multilayer network if no layers are specified)pbdoc");
    
    m.def("edges_idx", &edges_idx,
             py::arg("n"),
    R"pbdoc(Returns the list of edges, where vertex ids are used instead of vertex names)pbdoc");
    
    m.def("num_layers", &numLayers,
             py::arg("n"),
    R"pbdoc(Returns the number of layers in the input mlnetwork)pbdoc");
    
    m.def("num_actors", &numActors,
             py::arg("n"),
          py::arg("layers") = py::list(),
    R"pbdoc(Returns the number of actors in the set of input layers, or in the whole mlnetwork if no layers are specified)pbdoc");
    
    m.def("num_vertices", &numNodes,
             py::arg("n"),
          py::arg("layers") = py::list(),
    R"pbdoc(Returns the number of vertices in the set of input layers, or in the whole mlnetwork if no layers are specified)pbdoc");
    
    m.def("num_edges", &numEdges,
             py::arg("n"),
          py::arg("layers1") = py::list(),
          py::arg("layers2") = py::list(),
    R"pbdoc(Returns the number of edges in the set of input layers, or in the whole mlnetwork if no layers are specified)pbdoc");
    
    m.def("is_directed", &isDirected,
             py::arg("n"),
          py::arg("layers1") = py::list(),
          py::arg("layers2") = py::list(),
    R"pbdoc(Returns a logical vector indicating for each pair of layers if it is directed or not)pbdoc");
    
    /**************************************/
    /* NAVIGATION                         */
    /**************************************/
    
    
    m.def("neighbors", &actor_neighbors,
          py::arg("n"),
          py::arg("actor"),
          py::arg("layers") = py::list(),
          py::arg("mode") = "all",
    R"pbdoc(Returns the neighbors of a global identity on the set of input layers)pbdoc");
    
    m.def("xneighbors", &actor_xneighbors,
          py::arg("n"),
          py::arg("actor"),
          py::arg("layers") = py::list(),
          py::arg("mode") = "all",
    R"pbdoc(Returns the exclusive neighbors of a global identity on the set of input layers)pbdoc");
    
    
    /**************************************/
    /* MANIPULATION OF MULTILAYER NETWORKS */
    /**************************************/
    
    m.def("add_layers", &addLayers,
          py::arg("n"),
          py::arg("layers"),
          py::arg("directed") = py::list(),
    R"pbdoc(Adds one or more layers to a multilayer network)pbdoc");
    
    m.def("add_actors", &addActors,
          py::arg("n"),
          py::arg("actors"),
          R"pbdoc(Adds one or more actors to a multilayer network)pbdoc");
    
    m.def("add_vertices", &addNodes,
          py::arg("n"),
          py::arg("vertices"),
          R"pbdoc(Adds one or more vertices to a layer of a multilayer network)pbdoc");
    
    m.def("add_edges", &addEdges,
          py::arg("n"),
          py::arg("edges"),
          R"pbdoc(Adds one or more edges to a multilayer network - each edge is a quadruple [actor,layer,actor,layer])pbdoc");
    
    m.def("set_directed", &setDirected,
          py::arg("n"),
          py::arg("directionalities"),
          R"pbdoc(Set the directionality of one or more pairs of layers)pbdoc");
    
    m.def("delete_layers", &deleteLayers,
          py::arg("n"),
          py::arg("layers"),
          R"pbdoc(Deletes one or more layers from a multilayer network)pbdoc");
    
    m.def("delete_actors", &deleteActors,
          py::arg("n"),
          py::arg("actors"),
          R"pbdoc(Deletes one or more actors from a multilayer network)pbdoc");
    
    
    m.def("delete_vertices", &deleteNodes,
          py::arg("n"),
          py::arg("vertices"),
          R"pbdoc(Deletes one or more vertices from a layer of a multilayer network)pbdoc");
    
    
    m.def("delete_edges", &deleteEdges,
          py::arg("n"),
          py::arg("edges"),
          R"pbdoc(Deletes one or more edges from a multilayer network - each edge is a quadruple [actor,layer,actor,layer])pbdoc");
    
    
    /**************************************/
    /* ATTRIBUTES                         */
    /**************************************/
    
    m.def("add_attributes", &newAttributes,
          py::arg("n"),
          py::arg("attributes"),
          py::arg("type") = "string",
          py::arg("target") = "actor",
          py::arg("layer") = "",
          py::arg("layer1") = "",
          py::arg("layer2") = "",
          R"pbdoc(Creates a new attribute so that values can be associated to actors, layers, vertices or edges)pbdoc");
    
    m.def("attributes", &getAttributes,
          py::arg("n"),
          py::arg("target") = "actor",
          R"pbdoc(Returns the list of attributes defined for the input multilayer network)pbdoc");
    
    m.def("get_values", &getValues,
          py::arg("n"),
          py::arg("attribute"),
          py::arg("actors") = py::list(),
          py::arg("vertices") = py::dict(),
          py::arg("edges") = py::dict(),
          R"pbdoc(Returns the value of an attribute on the specified actors, layers, vertices or edges)pbdoc");
    
    m.def("set_values", &setValues,
          py::arg("n"),
          py::arg("attribute"),
          py::arg("actors") = py::list(),
          py::arg("vertices") = py::dict(),
          py::arg("edges") = py::dict(),
          py::arg("values"),
          R"pbdoc(Sets the value of an attribute for the specified actors/vertexes/edges)pbdoc");
    
    
    
    /**************************************/
    /* TRANSFORMATION                     */
    /**************************************/
    
    m.def("flatten", &flatten,
          py::arg("n"),
          py::arg("new.layer") = "flattening",
          py::arg("layers") = py::list(),
          py::arg("method") = "weighted",
          py::arg("force.directed") = false,
          py::arg("all.actors") = false,
          R"pbdoc(Adds a new layer with the actors in the input layers and an edge between A and B if they are connected in any of the merged layers)pbdoc");
    /*
     m.def("project", &project,
          py::arg("n"),
          py::arg("new.layer") = "projection",
          py::arg("layer1"),
          py::arg("layer2"),
          py::arg("method") = "clique",
          R"pbdoc(Adds a new layer with the actors in layer 1, and edges between actors A and B if they are connected to a common object in layer 2)pbdoc");
     
     // MEASURES
     */
    
    
    /**************************************/
    /* MEASURES                           */
    /**************************************/
    
    m.def("degree", &degree_ml,
          py::arg("n"),
          py::arg("actors") = py::list(),
          py::arg("layers") = py::list(),
          py::arg("mode") = "all",
          R"pbdoc(Returns the degree of each actor)pbdoc");
    
    m.def("degree_deviation", &degree_deviation_ml,
          py::arg("n"),
          py::arg("actors") = py::list(),
          py::arg("layers") = py::list(),
          py::arg("mode") = "all",
          R"pbdoc(Returns the standard deviation of the degree of each actor on the specified layers)pbdoc");
    /*
     m.def("occupation", &occupation,
          py::arg("n"),
          py::arg("transitions"),
          py::arg("teleportation") = .2,
          py::arg("steps") = 0,
          R"pbdoc(Returns the occupation centrality value of each actor)pbdoc");
     */
    m.def("neighborhood", &neighborhood_ml,
          py::arg("n"),
          py::arg("actors") = py::list(),
          py::arg("layers") = py::list(),
          py::arg("mode") = "all",
          R"pbdoc(Returns the neighborhood of each actor)pbdoc");
    
    m.def("xneighborhood", &xneighborhood_ml,
          py::arg("n"),
          py::arg("actors") = py::list(),
          py::arg("layers") = py::list(),
          py::arg("mode") = "all",
          R"pbdoc(Returns the exclusive neighborhood of each actor)pbdoc");
    
    m.def("connective_redundancy", &connective_redundancy_ml,
          py::arg("n"),
          py::arg("actors") = py::list(),
          py::arg("layers") = py::list(),
          py::arg("mode") = "all",
          R"pbdoc(Returns the connective redundancy of each actor)pbdoc");
    
    m.def("relevance", &relevance_ml,
          py::arg("n"),
          py::arg("actors") = py::list(),
          py::arg("layers") = py::list(),
          py::arg("mode") = "all",
          R"pbdoc(Returns the layer relevance of each actor)pbdoc");
    
    m.def("xrelevance", &xrelevance_ml,
          py::arg("n"),
          py::arg("actors") = py::list(),
          py::arg("layers") = py::list(),
          py::arg("mode") = "all",
          R"pbdoc(Returns the exclusive layer relevance of each actor)pbdoc");

    m.def("layer_summary", &summary_ml,
          py::arg("n"),
          py::arg("layer"),
          py::arg("method") = "entropy.degree",
          py::arg("mode") = "all",
          R"pbdoc(Computes a summary of the input layer)pbdoc");
    
    m.def("layer_comparison", &comparison_ml,
          py::arg("n"),
          py::arg("layers") = py::list(),
          py::arg("method") = "jaccard.edges",
          py::arg("mode") = "all",
          py::arg("K") = 0,
          R"pbdoc(Computes the similarity between the input layers)pbdoc");
    
    
    m.def("distance", &distance_ml,
          py::arg("n"),
          py::arg("from"),
          py::arg("to") = py::list(),
          py::arg("method") = "multiplex",
          R"pbdoc(Computes the distance between two actors)pbdoc");
    
    /**************************************/
    /* CLUSTERING                         */
    /**************************************/
    
    m.def("clique_percolation", &cliquepercolation_ml,
          py::arg("n"),
          py::arg("k") = 3,
          py::arg("m") = 1,
          R"pbdoc(Extension of the clique percolation method)pbdoc");
    
    m.def("glouvain", &glouvain_ml,
          py::arg("n"),
          py::arg("gamma") = 1,
          py::arg("omega") = 1,
          py::arg("limit") = 0,
          R"pbdoc(Extension of the louvain method)pbdoc");
    
    m.def("abacus", &abacus_ml,
          py::arg("n"),
          py::arg("min.actors") = 3,
          py::arg("min.layers") = 1,
          R"pbdoc(Community extraction based on frequent itemset mining)pbdoc");
    
    m.def("infomap", &infomap_ml,
          py::arg("n"),
                          py::arg("overlapping") = false,
                          py::arg("directed") = false,
                          py::arg("self.links") = true,
             R"pbdoc(Community extraction based on the flow equation)pbdoc");
    
    m.def("modularity", &modularity_ml,
                          py::arg("n"),
                          py::arg("comm.struct"),
                          py::arg("gamma") = 1,
                          py::arg("omega") = 1,
             R"pbdoc(Generalized modularity)pbdoc");
    
    /*
     m.def("lart", &lart,
          py::arg("n"),
          py::arg("t") = -1,
          py::arg("eps") = 1,
          py::arg("gamma") = 1,
          R"pbdoc(Community extraction based on locally adaptive random walks)pbdoc");
     */
    
    
    /**************************************/
    /* VISUALIZATION                      */
    /**************************************/
    
    m.def("layout_multiforce", &multiforce_ml,
          py::arg("n"),
          py::arg("w_in") = std::vector<double>({1}),
          py::arg("w_inter") = std::vector<double>({1}),
          py::arg("gravity") = std::vector<double>({0}),
          py::arg("iterations") = 100,
          R"pbdoc(Multiforce method: computes vertex coordinates)pbdoc");
    
    m.def("layout_circular", &circular_ml,
          py::arg("n") ,
          R"pbdoc(Circular method: computes vertex coordinates arranging actors on a circle)pbdoc");
    
        // plotting function defined in functions.R
    
    /*
    m.def("get_community_list", &to_list,
          py::arg("comm.struct"),
          py::arg("n"),
          R"pbdoc(Converts a community structure (data frame) into a list of communities, layer by layer)pbdoc");
     */
    
    
    /**************************************/
    /* PYTHON-SPECIFIC                    */
    /**************************************/

    m.def("to_edge_dict", &toNetworkxEdgeDict,
          py::arg("n") ,
          R"pbdoc(returns a representation of the edges on each layer compatible with networkx)pbdoc");
    
    m.def("to_node_dict", &toNetworkxNodeDict,
          py::arg("n") ,
          R"pbdoc(returns a representation of the actors on each layer compatible with networkx)pbdoc");
    
    
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
