#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "py_functions.hpp"
#include "utils/summary.hpp"

namespace py = pybind11;

using namespace py::literals;

PYBIND11_MODULE(_multinet, m) {
    m.doc() = R"pbdoc(
        Internal module used by uunet.multinet.
    )pbdoc";

    /*********************************************************************************/
    py::class_<PyMLNetwork>(m, "PyMLNetwork")
    .def(py::init<std::shared_ptr<uu::net::MultilayerNetwork>>())
    .def("__repr__",
         [](const PyMLNetwork& mnet) {
        return uu::net::summary_short(mnet.get_mlnet());
         }
         );
    
    /*********************************************************************************/
    py::class_<PyEvolutionModel>(m, "PyEvolutionModel")
    .def(py::init<std::shared_ptr<uu::net::EvolutionModel<uu::net::MultilayerNetwork>>,
         std::string>())
    .def("__repr__",
         [](const PyEvolutionModel& mod) {
             return mod.description();
         }
         );
    
    /*********************************************************************************/
    m.def("empty", &emptyMultilayer,
        py::arg("name") = "",
        R"pbdoc(
        Creates an empty network.
          
        Parameters
        __________
        name : str, optional
           Name of the new network (default: "")
        
        Returns
        _______
        n : PyMLNetwork
            A multilayer network.
        
        See Also
        ________
        add_actors
        add_layers
        add_vertices
        add_edges
          
        Examples
        ________
        >>> ml.empty()
        Multilayer Network [0 actors, 0 layers, 0 vertices, 0 edges (0,0)]
          
        )pbdoc");

    /*********************************************************************************/
    m.def("read", &readMultilayer,
        py::arg("file"),
        py::arg("name") = "unnamed",
        py::arg("aligned") = false,
        R"pbdoc(
        Reads a multilayer network from a file.
        
        Parameters
        ----------
        file : str
            The path of the file storing the multilayer network.
        name : str, optional
            The name of the multilayer network.
        aligned : bool, optional
            If True, all actors are added to all layers.
        
        Returns
        -------
        PyMLNetwork
            A multilayer network.
          
        Notes
        _____
        There are two network formats accepted: multiplex (default) or multilayer.
        A full multiplex network input file has the following format:
        
        -------------------------------------------------------------------------------------------
        -- comment lines start with two dashes (--)
        #VERSION
        3.0
        
        #TYPE
        multiplex
        
        #ACTOR ATTRIBUTES
        AttributeName1,STRING
        AttributeName2,NUMERIC
        -- etc.
        
        #NODE ATTRIBUTES
        LayerName1,AttributeName1,STRING
        LayerName1,AttributeName2,NUMERIC
        LayerName2,AttributeName3,STRING
        -- etc.
        
        #EDGE ATTRIBUTES
        -- edge attributes can be defined for specific layers (called local attributes):
        LayerName1,AttributeName,STRING
        LayerName1,AttributeName,NUMERIC
        -- or for all layers (called global attributes):
        AnotherAttributeName,NUMERIC
        -- etc.
        
        #LAYERS
        LayerName1,UNDIRECTED
        LayerName2,DIRECTED
        LayerName3,UNDIRECTED,LOOPS
        -- etc.
        -- LOOPS indicates that edges from one vertex to itself (loops) are allowed on that layer
        
        #ACTORS
        ActorName1,AttributeValueList...
        ActorName2,AttributeValueList...
        -- etc.
        
        #VERTICES
        ActorName1,LayerName1,AttributeValueList...
        ActorName1,LayerName2,AttributeValueList...
        -- etc.
        
        #EDGES
        ActorName1,ActorName2,LayerName1,LocalAttrValueList,GlobalAttrValueList...
        -- etc.
        -- the attribute values must be specified in the same order in which they are defined above
        -------------------------------------------------------------------------------------------
        
        If the #LAYERS section is empty, all edges are created as undirected.
      
        If the #ACTOR ATTRIBUTES, #VERTEX ATTRIBUTES or #EDGE ATTRIBUTES sections are empty,
        no attributes are created.
      
        The #LAYERS, #ACTORS and #VERTICES sections are useful only if attributes are present,
        or if there are actors that are not present in any layer (#ACTORS), or if there are
        isolated vertices (#VERTICES), otherwise they can be omitted.
      
        If no section is specified, #EDGES is the default.
      
        Therefore, a non attributed, undirected multiplex network file can be as simple as:
        
        -------------------------------------------------------------------------------------------
        Actor1,Actor2,Layer1
        Actor1,Actor3,Layer1
        Actor4,Actor2,Layer2
        -------------------------------------------------------------------------------------------
      
        If interlayer edges exist, then type "multilayer" must be specified,
        and layers and edges are formatted in a different way:
      
        -------------------------------------------------------------------------------------------
        #VERSION
        3.0
        
        #TYPE
        multilayer
        
        #ACTOR ATTRIBUTES
        AttributeName1,STRING
        AttributeName2,NUMERIC
        -- etc.
        
        #NODE ATTRIBUTES
        LayerName1,AttributeName1,STRING
        LayerName1,AttributeName2,NUMERIC
        LayerName2,AttributeName3,STRING
        -- etc.
        
        #EDGE ATTRIBUTES
        -- edge attributes can be defined for specific layers:
        LayerName1,AttributeName,STRING
        LayerName1,AttributeName,NUMERIC
        -- or for all layers (called global attributes):
        AnotherAttributeName,NUMERIC
        -- etc.
        
        #LAYERS
        -- LayerName1,LayerName1,UNDIRECTED
        -- LayerName2,LayerName2,DIRECTED
        -- LayerName3,LayerName3,DIRECTED,LOOPS
        -- LayerName1,LayerName2,DIRECTED
        -- etc.
        -- all intra-layer specifications (where the first and second layers are the same)
        -- should be listed first.
        -- LOOPS is only allowed for intra-layer specifications.
        
        #ACTORS
        ActorName1,AttributeValueList...
        ActorName2,AttributeValueList...
        -- etc.
        
        #VERTICES
        ActorName1,LayerName1,AttributeValueList...
        ActorName1,LayerName2,AttributeValueList...
        -- etc.
        
        #EDGES
        -- ActorName1,LayerName1,ActorName2,LayerName2,LocalAttrValueList,GlobalAttrValueList...
        -- etc.
        -------------------------------------------------------------------------------------------
        
        See Also
        ________
        data : predefined networks
        write : to file
        grow : synthetic networks
        )pbdoc");

    /*********************************************************************************/
    m.def("write", &writeMultilayer,
        py::arg("n"),
        py::arg("file"),
        py::arg("format") = "multilayer",
        py::arg("layers") = py::list(),
        py::arg("sep") = ',',
        py::arg("merge.actors") = true,
        py::arg("all.actors") = false,
        R"pbdoc(
        Writes a multilayer network to a file.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        file : str
            The path of the file storing the multilayer network.
        format : str
            Either "multilayer", to use the module's internal format, or "graphml".
        layers : list of str
            If specific layers are passed to the function, only those layers are saved to file.
        sep : char
            The character used in the file to separate text fields.
        merge.actors :
            Whether the nodes corresponding to each single actor should be merged into a
            single node (True) or kept separated (False), when format = "graphml" is used.
        all.actors :
            Whether all actors in the multilayer network should be included in the output
            file (True) or only those present in at least one of the input layers (False),
            when format = "graphml" and merge.actors = True are used.
        
        See Also
        ________
        read
        
        )pbdoc");
    
    /*********************************************************************************/
    m.def("evolution_pa", &ba_evolution_model,
        py::arg("m0"),
        py::arg("m"),
        R"pbdoc(
        Creates a layer evolutionary model based on preferential attachment.
        
        Parameters
        ----------
        m0 : int
            Initial number of nodes.
        m : int
            Number of edges created for each new vertex joining the network.
        
        Returns
        -------
        PyEvolutionModel
            An object instructing the grow() function how to modify a layer.
        
        See Also
        _______
        evolution_er
        grow
        
        )pbdoc");
    
    /*********************************************************************************/
    m.def("evolution_er", &er_evolution_model,
        py::arg("n"),
        R"pbdoc(
        Creates a layer evolutionary model based on random edge creation, as in the ER model.
        
        Parameters
        ----------
        n : int
            Number of nodes (created at the beginning, before starting adding edges).
        
        Returns
        -------
        PyEvolutionModel
          An object instructing the grow() function how to modify a layer.
        
        See Also
        ________
        evolution_pa
        grow
        )pbdoc");
    
    /*********************************************************************************/
    m.def("grow",&growMultiplex,
        py::arg("num.actors"),
        py::arg("num.steps"),
        py::arg("models"),
        py::arg("pr.internal"),
        py::arg("pr.external"),
        py::arg("dependency"),
        R"pbdoc(
        Grows a multiplex network.
        
        This function generates a multilayer network by letting it grow for a number of steps, where for each step three events can happen: (1) evolution according to internal dynamics (in which case a specific internal evolution model is used), (2) evolution importing edges from another layer, and (3) no action.
        
        The functions evolution_pa() and evolution_er() define, respectively, an evolutionary model based on preferential attachment and an evolutionary model where edges are created by choosing random end points, as in the ER random graph model.
        
        Parameters
        ----------
        num.actors : int
            The number of actors from which new nodes are selected during the generation process.
        num.steps : int
            Number of timestamps.
        models : list of PyEvolutionModel objects
            A vector containing one evolutionary model for each layer to be generated.
            Evolutionary models are defined using the evolution_*() functions.
        pr.internal : list of double
            A vector with (for each layer) the probability that at each step the layer evolves
            according to the internal evolutionary model.
        pr.external : list of double
            A vector with (for each layer) the probability that at each step the layer evolves
            importing edges from another layer.
        dependency : list of lists of double
            A matrix L x L where element (i,j) indicates the
            probability that layer i will import an edge from layer j in case an external
            event is triggered.
        
        Returns
        -------
        PyMLNetwork
        
        References
        __________
        Magnani, Matteo, and Luca Rossi (2013). Formation of Multiple Networks. In Social Computing, Behavioral-Cultural Modeling and Prediction, 257-264. Springer Berlin Heidelberg.
        
        See Also
        ________
        evolution_pa
        evolution_er
        data
        read
        )pbdoc");
    
    /*********************************************************************************/
    m.def("generate_communities", &generateCommunities,
        py::arg("type"),
        py::arg("num.actors"),
        py::arg("num.layers"),
        py::arg("num.communities"),
        py::arg("overlap") = 0,
          py::arg("pr.internal") = std::vector<double>({.4}),
          py::arg("pr.external") = std::vector<double>({.01}),
        R"pbdoc(
        Creates a network with a known community structure
        
        The generate_communities_ml function generates a simple community structure and a corresponding
        network with edges sampled according to that structure. Four simple models are available at the
        moment, all generating communities of equal size. In pillar community structures each actor belongs to
        the same community on all layers, while in semipillar community structures the communities in one
        layer are different from the other layers. In partitioning community structures each vertex belongs
        to one community, while in overlapping community structures some vertices belong to multiple
        communities. The four mode are: PEP (pillar partitioning), PEO (pillar overlapping),
          SEP (semipillar partitioning), SEO (semipillar overlapping).
        
        Parameters
        ----------
        type : str
            Type of community structure: pep, peo, sep or seo.
        num.actors : int
            The number of actors in the generated network.
        num.layers : int
            The number of layers in the generated network.
        overlap : int
            Number of actors at the end of one community to be also included in the following community.
        pr.internal : list of double
            A vector with the probability of adjacency for two vertices on the same layer
            and community (either a single value, or one value for each layer).
        pr.external : list of double
            A vector with the probability of adjacency for two vertices on the same layer
            but different communities (either a single value, or one value for each layer).
        
        Returns
        -------
        PyMLNetwork
        
        References
        __________
        Matteo Magnani, Obaida Hanteer, Roberto Interdonato, Luca Rossi, and Andrea Tagarelli (2021).
        Community Detection in Multiplex Networks.
        ACM Computing Surveys.

        See Also
        ________
        grow
        data
        read
        )pbdoc");

    
    /**************************************/
    /* INFORMATION ON MULTILAYER NETWORKS */
    /**************************************/
    
    /*********************************************************************************/
    m.def("layers",
         &layers,
         py::arg("n"),
        R"pbdoc(
        Returns the list of layers in the input multilayer network.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        
        Returns
        -------
        list
          A list of layer names.
        
        See Also
        ________
        )pbdoc");
    
    /*********************************************************************************/
    m.def("actors", &actors,
        py::arg("n"),
        py::arg("layers") = py::list(),
        py::arg("attributes") = false,
        R"pbdoc(
        Returns the list of actors present in the input layers, or in the whole multilayer network if no layers are specified.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        layers :
            A list of names of layers belonging to the network. Only the actors in these
            layers are returned. If the array is empty, all the actors in the network are
            returned.
        add_attributes :
            If true, actor attribute values are added to the result.
        
        
        Returns
        -------
        dict
          A dictionary with a list of actor names ("actor").
        
        See Also
        ________
        vertices, edges, layers
        )pbdoc");
    
    /*********************************************************************************/
    m.def("vertices", &vertices,
        py::arg("n"),
        py::arg("layers") = py::list(),
        py::arg("attributes") = false,
        R"pbdoc(
        Returns the list of vertices in the input layers, or in the whole multilayer network if no layers are specified.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        layers :
            A list of names of layers belonging to the network. Only the vertices in these
            layers are returned. If the array is empty, all the vertices in the network are
            returned.
        add_attributes :
            If true, attribute values are added to the result.
        
        Returns
        -------
        dict
          With two lists: "actor" and "layer"
        
        See Also
        ________
        )pbdoc");
    
    /*********************************************************************************/
    m.def("edges", &edges,
        py::arg("n"),
        py::arg("layers1") = py::list(),
        py::arg("layers2") = py::list(),
        py::arg("attributes") = false,
        R"pbdoc(
        Returns the list of edges among vertices in the input layers (if only one set of layers
        is specified), or from the first set of input layers to the second set of input layers,
        or in the whole multilayer network if no layers are specified.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        layers1 : list of str
            The layer(s) from where the edges to be extracted start. If an empty list of layers
            is passed (default), all the layers are considered.
        layers2 : list of str
            The layer(s) where the edges to be extracted end. If an empty list of layers is
            passed (default), the ending layers are set as equal to those in parameter layer1.
        add_attributes :
            If true, actor attribute values are added to the result.
        
        Returns
        -------
        dict
            With five lists: "actor_from", "layer_from", "actor_to", "layer_to", "dir".
        
        See Also
        ________
        )pbdoc");
    
    /*********************************************************************************/
    m.def("edges_idx", &edges_idx,
         py::arg("n"),
        R"pbdoc(
        Returns the list of edges, as in edges(), but with vertex ids instead of vertex
        and layer names.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        
        Returns
        -------
        dict
        
        See Also
        ________
        )pbdoc");
    
    /*********************************************************************************/
    m.def("num_layers", &numLayers,
         py::arg("n"),
        R"pbdoc(
        Returns the number of layers in the input mlnetwork.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        
        Returns
        -------
        int
        
        See Also
        ________
        actors
        layers
        vertices
        edges
        num_actors
        num_vertices
        num_edges
        is_directed
        )pbdoc");
    
    /*********************************************************************************/
    m.def("num_actors", &numActors,
         py::arg("n"),
        py::arg("layers") = py::list(),
        R"pbdoc(
        Returns the number of actors in the set of input layers, or in the whole mlnetwork if
        no layers are specified.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        
        Returns
        -------
        int
        
        See Also
        ________
        actors
        layers
        vertices
        edges
        num_layers
        num_vertices
        num_edges
        is_directed
        )pbdoc");
    
    /*********************************************************************************/
    m.def("num_vertices", &numNodes,
         py::arg("n"),
        py::arg("layers") = py::list(),
        R"pbdoc(
        Returns the number of vertices in the set of input layers, or in the whole network
        if no layers are specified.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        layers : list of str
            A list of names of layers belonging to the network. Only the actors/vertices in these
            layers are returned. If the array is empty, all the vertices in the network are
            returned. Notice that this may not correspond to the list of actors: there can be
            actors that are not present in any layer. These would be returned only using the
            actors() function.
        
        Returns
        -------
        int
        
        See Also
        ________
        actors
        layers
        vertices
        edges
        num_actors
        num_layers
        num_edges
        is_directed
        )pbdoc");
    
    /*********************************************************************************/
    m.def("num_edges", &numEdges,
         py::arg("n"),
        py::arg("layers1") = py::list(),
        py::arg("layers2") = py::list(),
        R"pbdoc(
        Returns the number of edges in the set of input layers, or in the whole network if no layers are specified.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        layers1 : list of str
            The layer(s) from where the edges to be extracted start. If an empty list of layers
            is passed (default), all the layers are considered.
        layers2 : list of str
            The layer(s) where the edges to be extracted end. If an empty list of layers is
            passed (default), the ending layers are set as equal to those in parameter layer1.
        
        Returns
        -------
        int
        
        See Also
        ________
        actors
        layers
        vertices
        edges
        num_actors
        num_layers
        num_vertices
        is_directed
        )pbdoc");
    
    /*********************************************************************************/
    m.def("is_directed", &isDirected,
         py::arg("n"),
        py::arg("layers1") = py::list(),
        py::arg("layers2") = py::list(),
        R"pbdoc(
        Returns a logical vector indicating for each pair of layers if it is directed or not.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        
        Returns
        -------
        bool
        
        See Also
        ________
        actors
        layers
        vertices
        edges
        num_actors
        num_layers
        num_vertices
        num_edges
        )pbdoc");
    
    /**************************************/
    /* NAVIGATION                     */
    /**************************************/
    
    /*********************************************************************************/
    m.def("neighbors", &actor_neighbors,
        py::arg("n"),
        py::arg("actor"),
        py::arg("layers") = py::list(),
        py::arg("mode") = "all",
        R"pbdoc(
        Returns the neighbors of a global identity on the set of input layers.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        actor : str
            An actor name present in the network, whose neighbors are extracted.
        layers : list of str
            An array of layers belonging to the network. Only the nodes in these layers are
            returned. If the array is empty, all the nodes in the network are returned.
        mode : str
            This argument can take values "in", "out" or "all" to indicate respectively neighbors
            reachable via incoming edges, via outgoing edges or both.
        
        Returns
        -------
        list
            A list of actor names who are connected to the input actor on at least one of
            the specified layers.
        
        References
        __________
        Berlingerio, Michele, Michele Coscia, Fosca Giannotti, Anna Monreale, and
        Dino Pedreschi (2011). "Foundations of Multidimensional Network Analysis."
        In International Conference on Social Network Analysis and Mining (ASONAM), 485-89.
        IEEE Computer Society.
        
        See Also
        ________
        xneighbors
        )pbdoc");
    
    /*********************************************************************************/
    m.def("xneighbors", &actor_xneighbors,
        py::arg("n"),
        py::arg("actor"),
        py::arg("layers") = py::list(),
        py::arg("mode") = "all",
        R"pbdoc(
        Returns the exclusive neighbors of a global identity on the set of input layers.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        actor : str
            An actor name present in the network, whose neighbors are extracted.
        layers : list of str
            An array of layers belonging to the network. Only the nodes in these layers are
            returned. If the array is empty, all the nodes in the network are returned.
        mode : str
            This argument can take values "in", "out" or "all" to indicate respectively
            neighbors reachable via incoming edges, via outgoing edges or both.
        
        Returns
        -------
        list
          A list of actor names who are connected to the input actor on at least one of the specified layers, and on none of the other layers. Exclusive neighbors are those neighbors that would be lost by removing the input layers.
        
        References
        __________
        Berlingerio, Michele, Michele Coscia, Fosca Giannotti, Anna Monreale, and
        Dino Pedreschi (2011). "Foundations of Multidimensional Network Analysis."
        In International Conference on Social Network Analysis and Mining (ASONAM), 485-89.
        IEEE Computer Society.
        
        See Also
        ________
        xneighbors
        )pbdoc");
    
    
    /**************************************/
    /* MANIPULATION OF MULTILAYER NETWORKS */
    /**************************************/
    
    /*********************************************************************************/
    m.def("add_layers", &addLayers,
        py::arg("n"),
        py::arg("layers"),
        py::arg("directed") = py::list(),
        R"pbdoc(
        Adds one or more layers to a multilayer network.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        layers : list of str
            Names of layers.
        directed : bool
            Determines if the layer(s) is (are) directed or undirected. If multiple layers
            are specified, directed should be either a single value or an array with as many
            values as the number of layers.

        See Also
        ________
        add_actors
        add_vertices
        add_edges
        set_directed
        delete_layers
        delete_actors
        delete_vertices
        delete_edges
        )pbdoc");
    
    /*********************************************************************************/
    m.def("add_actors", &addActors,
        py::arg("n"),
        py::arg("actors"),
        R"pbdoc(
        Adds one or more actors to a multilayer network.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        actors : list of str
            Names of actors.
        
        See Also
        ________
        add_layers
        add_vertices
        add_edges
        set_directed
        delete_layers
        delete_actors
        delete_vertices
        delete_edges
        )pbdoc");
    
    /*********************************************************************************/
    m.def("add_vertices", &addNodes,
        py::arg("n"),
        py::arg("vertices"),
        R"pbdoc(
        Adds one or more vertices to a layer of a multilayer network.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        vertices : dict
            A dictionary of vertices to be updated. The list "actor" specifies actor names,
            the list "layer" layer names.
        
        See Also
        ________
        add_layers
        add_actors
        add_edges
        set_directed
        delete_layers
        delete_actors
        delete_vertices
        delete_edges
        )pbdoc");
    
    /*********************************************************************************/
    m.def("add_edges", &addEdges,
        py::arg("n"),
        py::arg("edges"),
        R"pbdoc(
        Adds one or more edges to a multilayer network - each edge is a quadruple
        [actor,layer,actor,layer].
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        edges : dict
            A dictionary containing the vertices to be connected. The four lists must contain:
            "from_actor" names, "from_layer" names, "to_actor" names, "to_layer" names.
            The directionality of the edge (directed/undirected) is pre-defined depending
            on the layer(s).
        
        See Also
        ________
        add_layers
        add_actors
        add_vertices
        set_directed
        delete_layers
        delete_actors
        delete_vertices
        delete_edges
        )pbdoc");
    
    /*********************************************************************************/
    m.def("set_directed", &setDirected,
        py::arg("n"),
        py::arg("directionalities"),
        R"pbdoc(
        Set the directionality of one or more pairs of layers.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        directionalities : dict
            "layer1" (str), "layer2" (str), "dir" (bool)
          
        See Also
        ________
        add_layers
        add_actors
        add_vertices
        add_edges
        delete_layers
        delete_actors
        delete_vertices
        delete_edges
        )pbdoc");
    
    /*********************************************************************************/
    m.def("delete_layers", &deleteLayers,
        py::arg("n"),
        py::arg("layers"),
        R"pbdoc(
        Deletes one or more layers from a multilayer network.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        layers : list of str
            Names of layers.
        
        See Also
        ________
        add_layers
        add_actors
        add_vertices
        add_edges
        set_directed
        delete_actors
        delete_vertices
        delete_edges
        )pbdoc");
    
    /*********************************************************************************/
    m.def("delete_actors", &deleteActors,
        py::arg("n"),
        py::arg("actors"),
        R"pbdoc(
        Deletes one or more actors from a multilayer network.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        actors : list of str
            Names of actors.
        
        See Also
        ________
        add_layers
        add_actors
        add_vertices
        add_edges
        set_directed
        delete_layers
        delete_vertices
        delete_edges
        )pbdoc");
    
    
    /*********************************************************************************/
    m.def("delete_vertices", &deleteNodes,
        py::arg("n"),
        py::arg("vertices"),
        R"pbdoc(
        Deletes one or more vertices from a layer of a multilayer network.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        vertices : dict
            A dictionary of vertices to be deleted. The list "actor" specifies actor names,
            the list "layer" layer names.
        
        See Also
        ________
        add_layers
        add_actors
        add_vertices
        add_edges
        set_directed
        delete_layers
        delete_actors
        delete_edges
        )pbdoc");
    
    
    /*********************************************************************************/
    m.def("delete_edges", &deleteEdges,
        py::arg("n"),
        py::arg("edges"),
        R"pbdoc(
        Deletes one or more edges from a multilayer network - each edge is a quadruple [actor,layer,actor,layer].
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        edges : dict
            A dictionary containing the edges to be deleted. The four lists must contain:
            "from_actor" names, "from_layer" names, "to_actor" names, "to_layer" names.
        
        See Also
        ________
        add_layers
        add_actors
        add_vertices
        add_edges
        set_directed
        delete_layers
        delete_actors
        delete_vertices
        )pbdoc");
    
    
    /**************************************/
    /* ATTRIBUTES                     */
    /**************************************/
    
    /*********************************************************************************/
    m.def("add_attributes", &newAttributes,
        py::arg("n"),
        py::arg("attributes"),
        py::arg("type") = "string",
        py::arg("target") = "actor",
        py::arg("layer") = "",
        py::arg("layer1") = "",
        py::arg("layer2") = "",
        R"pbdoc(
        Creates a new attribute so that values can be associated to actors, layers, vertices or edges.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        attributes :
            Name(s) of the attributes to be created.
        type : str
            Can be "string" or "numeric".
        target : str
            Can be "actor" (attributes attached to actors), "vertex" (attributes attached to
            vertices) or "edge" (attributes attached to edges). Layer attributes are not available
            in this version.
        layer : str
            This can be specified only for targets "vertex" (so that the attribute exists only
            for the vertices in that layer) or "edge" (in which case the attribute applies to
            intra-layer edges in that layer).
        layer1 : str
            This can be specified only for target "edge", together with layer2, so that the
            attribute applies to inter-layer edges from layer1 to layer2. If layer1 and
            layer2 are specified, the parameter layer should not be used.
        layer2 : str
            See layer1.
        
        See Also
        ________
        attributes
        get_values
        set_values
        )pbdoc");
    
    /*********************************************************************************/
    m.def("attributes", &getAttributes,
        py::arg("n"),
        py::arg("target") = "actor",
        R"pbdoc(
        Returns the list of attributes defined for the input multilayer network.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        target : str
            Can be "actor" (attributes attached to actors), "vertex" (attributes attached to
            vertices) or "edge" (attributes attached to edges). Layer attributes are not
            available in this version.

        Returns
        -------
        dict
          Containing attribute name and types, in additio to layer information for targets vertex
          and edge.
        
        See Also
        ________
        add_attributes
        get_values
        set_values
        )pbdoc");
    
    /*********************************************************************************/
    m.def("get_values", &getValues,
        py::arg("n"),
        py::arg("attribute"),
        py::arg("actors") = py::dict(),
        py::arg("vertices") = py::dict(),
        py::arg("edges") = py::dict(),
        R"pbdoc(
        Returns the value of an attribute on the specified actors, layers, vertices or edges.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        attribute : str
            The name of the attribute to be updated.
        actors : dict
            A dictionary containing a list of actor names called "actor". If this is specified, layers, vertices and edges should not.
        vertices : dict
            Vertices to be updated. The first column specifies actor names,
            the second layer names. If this is specified, actors, layers and edges should not.
        edges : dict
          Vertices to be connected. The four lists must contain:
          "from_actor" names, "from_layer" names, "to_actor" names, "to_layer" names.
        
        Returns
        -------
        dict
          Containing one list with attribute values.
        
        See Also
        ________
        add_attributes
        attributes
        set_values
        )pbdoc");
    
    /*********************************************************************************/
    m.def("set_values", &setValues,
        py::arg("n"),
        py::arg("attribute"),
        py::arg("actors") = py::dict(),
        py::arg("vertices") = py::dict(),
        py::arg("edges") = py::dict(),
        py::arg("values"),
        R"pbdoc(
        Sets the value of an attribute for the specified actors/vertexes/edges.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        attribute : str
            The name of the attribute to be updated.
        actors : list of str
            A vector of actor names. If this is specified, layers, vertices and edges should not.
        vertices : list of str
            Vertices to be updated. The first column specifies actor names, the second layer names.
            If this is specified, actors, layers and edges should not.
        edges : dict
          Vertices to be connected. The four lists must contain:
          "from_actor" names, "from_layer" names, "to_actor" names, "to_layer" names.
        values : list
            A vector of values to be set for the corresponding actors, vertices or edges.
        

        See Also
        ________
        add_attributes
        attributes
        get_values
        )pbdoc");
    
    
    
    /**************************************/
    /* TRANSFORMATION                 */
    /**************************************/
    
    /*********************************************************************************/
    m.def("flatten", &flatten,
        py::arg("n"),
        py::arg("new.layer") = "flattening",
        py::arg("layers") = py::list(),
        py::arg("method") = "weighted",
        py::arg("force.directed") = false,
        py::arg("all.actors") = false,
        R"pbdoc(
        Adds a new layer with the actors in the input layers and an edge between A and B
        if they are connected in any of the merged layers.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        new.layer : str
            Name of the new layer.
        layers : list of str
            An array of layers belonging to the network.
        method : str
            This argument can take values "weighted" or "or". "weighted" adds an attribute to
            the new edges with the number of layers where the two actors are connected}
        force.directed : bool
            The new layer is set as directed. If this is False, the new layer is set as directed
            if at least one of the merged layers is directed.}
        all.actors :
            If True, then all the actors are included in the new layer, even if they are not
            present in any of the merged layers.}
        
        See Also
        ________
        )pbdoc");

    /*********************************************************************************/
    m.def("project", &project,
        py::arg("n"),
        py::arg("new.layer") = "projection",
        py::arg("layer1"),
        py::arg("layer2"),
        py::arg("method") = "clique",
        R"pbdoc(
        Adds a new layer with the actors in layer 1, and edges between actors A and B if they are connected to a common object in layer 2.
          
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        new.layer : str
            Name of the new layer.
        layer1 : str
            Name of the layer from which actors are taken.
        layer2 : str
            Name of the layer to be projected on layer1.
        method : str
            Currently only the "clique" method is implemented, creating an edge between A and be
            if they are adjacent to at least one common object on layer2.
          
        See Also
        ________
        )pbdoc");

    /**************************************/
    /* MEASURES                       */
    /**************************************/
    
    /*********************************************************************************/
    m.def("degree", &degree_ml,
        py::arg("n"),
        py::arg("actors") = py::list(),
        py::arg("layers") = py::list(),
        py::arg("mode") = "all",
        R"pbdoc(
        Returns the degree of each actor.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        actors : list of str
            Actor names.
        layers : list of str
            Layer names.
        mode : str
            This argument can take values "in", "out" or "all" to count respectively incoming
            edges, outgoing edges or both.
        
        Returns
        -------
        list of int
        
        See Also
        ________
        degree_deviation
        neighborhood
        xneighborhood
        connective_redundancy
        relevance
        xrelevance
        )pbdoc");
    
    /*********************************************************************************/
    m.def("degree_deviation", &degree_deviation_ml,
        py::arg("n"),
        py::arg("actors") = py::list(),
        py::arg("layers") = py::list(),
        py::arg("mode") = "all",
        R"pbdoc(
        Returns the standard deviation of the degree of each actor on the specified layers.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        actors : list of str
            Actor names.
        layers : list of str
            Layer names.
        mode : str
            This argument can take values "in", "out" or "all" to count respectively incoming
            edges, outgoing edges or both.
        
        Returns
        -------
        list of double
        
        See Also
        ________
        degree
        neighborhood
        xneighborhood
        connective_redundancy
        relevance
        xrelevance
        )pbdoc");
    
    /*********************************************************************************/
    /*m.def("occupation", &occupation,
        py::arg("n"),
        py::arg("transitions"),
        py::arg("teleportation") = .2,
        py::arg("steps") = 0,
        R"pbdoc(
        Returns the occupation centrality value of each actor)pbdoc");
     */
    /*********************************************************************************/
    m.def("neighborhood", &neighborhood_ml,
        py::arg("n"),
        py::arg("actors") = py::list(),
        py::arg("layers") = py::list(),
        py::arg("mode") = "all",
        R"pbdoc(
        Returns the neighborhood of each actor.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        actors : list of str
            Actor names.
        layers : list of str
            Layer names.
        mode : str
            This argument can take values "in", "out" or "all" to count respectively incoming
            edges, outgoing edges or both.
        
        Returns
        -------
        list of int
        
        See Also
        ________
        degree
        degree_deviation
        xneighborhood
        connective_redundancy
        relevance
        xrelevance
        )pbdoc");
    
    /*********************************************************************************/
    m.def("xneighborhood", &xneighborhood_ml,
        py::arg("n"),
        py::arg("actors") = py::list(),
        py::arg("layers") = py::list(),
        py::arg("mode") = "all",
        R"pbdoc(
        Returns the exclusive neighborhood of each actor.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        actors : list of str
            Actor names.
        layers : list of str
            Layer names.
        mode : str
            This argument can take values "in", "out" or "all" to count respectively incoming
            edges, outgoing edges or both.
        
        Returns
        -------
        list of int
        
        See Also
        ________
        degree
        degree_deviation
        neighborhood
        connective_redundancy
        relevance
        xrelevance
        )pbdoc");
    
    /*********************************************************************************/
    m.def("connective_redundancy", &connective_redundancy_ml,
        py::arg("n"),
        py::arg("actors") = py::list(),
        py::arg("layers") = py::list(),
        py::arg("mode") = "all",
        R"pbdoc(
        Returns the connective redundancy of each actor.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        actors : list of str
            Actor names.
        layers : list of str
            Layer names.
        mode : str
            This argument can take values "in", "out" or "all" to count respectively incoming
            edges, outgoing edges or both.
        
        Returns
        -------
        list of double
        
        See Also
        ________
        degree
        degree_deviation
        neighborhood
        xneighborhood
        relevance
        xrelevance
        )pbdoc");
    
    /*********************************************************************************/
    m.def("relevance", &relevance_ml,
        py::arg("n"),
        py::arg("actors") = py::list(),
        py::arg("layers") = py::list(),
        py::arg("mode") = "all",
        R"pbdoc(
        Returns the layer relevance of each actor.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        actors : list of str
            Actor names.
        layers : list of str
            Layer names.
        mode : str
            This argument can take values "in", "out" or "all" to count respectively incoming
            edges, outgoing edges or both.
        
        Returns
        -------
        list of double
        
        See Also
        ________
        degree
        degree_deviation
        neighborhood
        xneighborhood
        connective_redundancy
        xrelevance
        )pbdoc");
    
    /*********************************************************************************/
    m.def("xrelevance", &xrelevance_ml,
        py::arg("n"),
        py::arg("actors") = py::list(),
        py::arg("layers") = py::list(),
        py::arg("mode") = "all",
        R"pbdoc(
        Returns the exclusive layer relevance of each actor.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        actors : list of str
            Actor names.
        layers : list of str
            Layer names.
        mode : str
            This argument can take values "in", "out" or "all" to count respectively incoming
            edges, outgoing edges or both.}
        
        Returns
        -------
        list of double
        
        See Also
        ________
        degree
        degree_deviation
        neighborhood
        xneighborhood
        connective_redundancy
        relevance
        )pbdoc");

    /*********************************************************************************/
    m.def("layer_summary", &summary_ml,
        py::arg("n"),
        py::arg("layer"),
        py::arg("method") = "entropy.degree",
        py::arg("mode") = "all",
        R"pbdoc(
        Computes a summary of the input layer.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        layer : str
            The name of a layer.
        method : str
            This argument can take the following values: "min.degree", "max.degree", "sum.degree", "mean.degree", "sd.degree", "skewness.degree", "kurtosis.degree", "entropy.degree", "CV.degree", "jarque.bera.degree".
        mode : str
            This argument is used for distribution dissimilarities and correlations (that is,
            those methods based on node degree) and can take values "in", "out" or "all"
            to consider respectively incoming edges, outgoing edges or both.
        
        Returns
        -------
        double
        
        See Also
        ________
        layer_comparison
        )pbdoc");
    
    /*********************************************************************************/
    m.def("layer_comparison", &comparison_ml,
        py::arg("n"),
        py::arg("layers") = py::list(),
        py::arg("method") = "jaccard.edges",
        py::arg("mode") = "all",
        py::arg("K") = 0,
        R"pbdoc(
        Computes the similarity between the input layers.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        layer : str
            The name of a layer.
        layers : list of str
            Names of the layers to be compared. If not specified, all layers are used.}
        method : str
            This argument can take the following values.
            - For overlapping-based measures: "jaccard.actors", "jaccard.edges", "jaccard.triangles",
            "coverage.actors", "coverage.edges",  "coverage.triangle","sm.actors", "sm.edges",
            "sm.triangles", "rr.actors", "rr.edges", "rr.triangles", "kulczynski2.actors",
            "kulczynski2.edges", "kulczynski2.triangles", "hamann.actors", "hamann.edges",
            "hamann.triangles". The first part of the value indicates the type of comparison
            function (Jaccard, Coverage, Simple Matching, Russell Rao, Kulczynski, Hamann),
            the second part indicates the configurations to which the comparison function
            is applied.
            - For distribution dissimilarities: "dissimilarity.degree", "KL.degree", "jeffrey.degree".
            Notice that these are dissimilarity functions: 0 means highest similarity.
            - For correlations: "pearson.degree" and "rho.degree"
        mode : str
            This argument is used for distribution dissimilarities and correlations (that is,
            those methods based on node degree) and can take values "in", "out" or "all" to
            consider respectively incoming edges, outgoing edges or both.
        K : int
            This argument is used for distribution dissimilarity measures and indicates
            the number of histogram bars used to compute the divergence. If 0 is specified,
            then a "typical" value is used, close to the logarithm of the number of actors.
        
        Returns
        -------
        list of lists of double
            Layer-by-layer comparisons: for each pair of layers, a value between 0 and 1
            (for overlapping and distribution dissimilarity) or -1 and 1 (for correlation).
            
        References
        __________
        Brodka, P., Chmiel, A., Magnani, M., and Ragozini, G. (2018). Quantifying layer similarity in multiplex networks: a systematic study. Royal Sociwty Open Science 5(8)
            
        See Also
        ________
        layer_summary
        )pbdoc");
    
    
    /*********************************************************************************/
    m.def("distance", &distance_ml,
        py::arg("n"),
        py::arg("from"),
        py::arg("to") = py::list(),
        py::arg("method") = "multiplex",
        R"pbdoc(
        Computes the distance between two actors.
        
        This function is based on the concept of multilayer distance. This concept generalizes
        single-layer distance to a vector with the distance traveled on each layer (in the
        "multiplex" case). Therefore, non-dominated path lengths are returned instead of shortest
        path length, where one path length dominates another if it is not longer on all layers,
        and shorter on at least one. A non-dominated path length is also known as a Pareto distance.
        Finding all multilayer distances can be very time-consuming for large networks.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        from : str
            The actor from which the distance is computed.
        to : list of str
            The actor(s) to which the distance is computed. If not specified, all actors are
            considered.
        method :
            This argument can take values "simple", "multiplex", "full". Only "multiplex"
            is currently available in the python module.
        
        Returns
        -------
        dict
          With one list for each layer specifying the number of steps in that layer.
        
        References
        __________
        Magnani, Matteo, and Rossi, Luca (2013). Pareto Distance for Multi-layer Network Analysis.
        In Social Computing, Behavioral-Cultural Modeling and Prediction (Vol. 7812, pp. 249-256).
        Springer Berlin Heidelberg.
        )pbdoc");
    
    /**************************************/
    /* CLUSTERING                     */
    /**************************************/
    
    /*********************************************************************************/
    m.def("clique_percolation", &cliquepercolation_ml,
        py::arg("n"),
        py::arg("k") = 3,
        py::arg("m") = 1,
        R"pbdoc(
        Extension of the clique percolation method.
        
        All directed edges are considered as undirected.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        k : int
            Minimum number of actors in a clique. Must be at least 3.
        m : int
            Minimum number of common layers in a clique. Not to be confused with number of edges, as it is meant in the summary function (here we use the notation of the paper introducing this algorithm).

        Returns
        -------
        dict
          "actor", "layer", "cid" (community id).
        
        References
        __________
        Afsarmanesh, Nazanin, and Magnani, Matteo (2018). Partial and overlapping community detection in multiplex social networks. Social informatics.
        
        See Also
        ________
        abacus
        glouvain
        infomap
        flat_ec
        flat_nw
        mdlp
        modularity
        )pbdoc");
    
    /*********************************************************************************/
    m.def("glouvain", &glouvain_ml,
        py::arg("n"),
        py::arg("gamma") = 1.0,
        py::arg("omega") = 1.0,
        R"pbdoc(
        Extension of the louvain method.
        
        It only works on undirected networks, and considers weights if all layers have an edge attribute named w_ of type DOUBLE.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        gamma : double
            Resolution parameter.
        omega : double
            Inter-layer weight parameter in the generalized louvain method.

        Returns
        -------
        dict
        "actor", "layer", "cid" (community id).
        
        References
        __________
        Mucha, Peter J., Richardson, Thomas, Macon, Kevin, Porter, Mason A., and Onnela, Jukka-Pekka (2010). Community structure in time-dependent, multiscale, and multiplex networks. Science (New York, N.Y.), 328(5980), 876-8. Data Analysis, Statistics and Probability; Physics and Society.
        
        See Also
        ________
        abacus
        clique_percolation
        infomap
        flat_ec
        flat_nw
        mdlp
        modularity
        )pbdoc");
    
    /*********************************************************************************/
    m.def("abacus", &abacus_ml,
        py::arg("n"),
        py::arg("min.actors") = 3,
        py::arg("min.layers") = 1,
        R"pbdoc(
        Community extraction based on frequent itemset mining.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        min.actors : int
            Minimum number of actors to form a community.
        min.layers : int
            Minimum number of times two actors must be in the same single-layer community to be
            considered in the same multi-layer community.

        Returns
        -------
        dict
        "actor", "layer", "cid" (community id).
        
        References
        __________
        Berlingerio, Michele, Pinelli, Fabio, and Calabrese, Francesco (2013).
        ABACUS: frequent pAttern mining-BAsed Community discovery in mUltidimensional networkS.
        Data Mining and Knowledge Discovery, 27(3), 294-320.
        
        See Also
        ________
        clique_percolation
        glouvain
        infomap
        flat_ec
        flat_nw
        mdlp
        modularity
        )pbdoc");
    
    /*********************************************************************************/
    m.def("infomap", &infomap_ml,
        py::arg("n"),
        py::arg("overlapping") = false,
        py::arg("directed") = false,
        py::arg("self_links") = true,
        R"pbdoc(
        Community extraction based on the flow equation.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        overlapping : bool
            Specifies if overlapping clusters can be returned.
        directed : bool
            Specifies whether the edges should be considered as directed.
        self.links : bool
            Specifies whether self links should be considered or not.

        Returns
        -------
        dict
        "actor", "layer", "cid" (community id).
        
        References
        __________
        De Domenico, M., Lancichinetti, A., Arenas, A., and Rosvall, M. (2015)
        Identifying Modular Flows on Multilayer Networks Reveals Highly Overlapping Organization
        in Interconnected Systems. PHYSICAL REVIEW X 5, 011027.
        
        See Also
        ________
        abacus
        clique_percolation
        glouvain
        flat_ec
        flat_nw
        mdlp
        modularity
        )pbdoc");
    
    
    /*********************************************************************************/
    m.def("flat_ec", &flat_ec,
        py::arg("n"),
        R"pbdoc(
        Community extraction based on flattening (weighted, edge count).
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.

        Returns
        -------
        dict
        "actor", "layer", "cid" (community id).
        
        References
        __________
        Michele Berlingerio, Michele Coscia, and Fosca Giannotti.
        Finding and characterizing communities in multidimensional networks.
        In International Conference on Advances in Social Networks Analysis and Mining (ASONAM), pages 490-494. IEEE Computer Society Washington, DC, USA, 2011
        
        See Also
        ________
        clique_percolation
        abacus
        glouvain
        infomap
        flat_ec
        mdlp
        modularity
        )pbdoc");
    
    /*********************************************************************************/
    m.def("flat_nw", &flat_nw,
        py::arg("n"),
        R"pbdoc(
        Community extraction based on flattening (not weighted).
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.

        Returns
        -------
        dict
        "actor", "layer", "cid" (community id).
        
        References
        __________
        Michele Berlingerio, Michele Coscia, and Fosca Giannotti.
        Finding and characterizing communities in multidimensional networks.
        In International Conference on Advances in Social Networks Analysis and Mining (ASONAM), pages 490-494. IEEE Computer Society Washington, DC, USA, 2011
        
        See Also
        ________
        clique_percolation
        abacus
        glouvain
        infomap
        flat_ec
        mdlp
        modularity
        )pbdoc");
    
    /*********************************************************************************/
    m.def("mdlp", &mdlp,
        py::arg("n"),
        R"pbdoc(
        Community extraction based on label propagation.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.

        Returns
        -------
        dict
        "actor", "layer", "cid" (community id).
        
        References
        __________
        Oualid Boutemine and Mohamed Bouguessa.
        Mining Community Structures in Multidimensional Networks.
        ACM Transactions on Knowledge Discovery from Data, 11(4):1-36, 2017
        
        See Also
        ________
        clique_percolation
        abacus
        glouvain
        infomap
        flat_ec
        flat_nw
        modularity
        )pbdoc");
    
    
    /*********************************************************************************/
    m.def("modularity", &modularity_ml,
        py::arg("n"),
        py::arg("comm.struct"),
        py::arg("gamma") = 1,
        py::arg("omega") = 1,
        R"pbdoc(
        Generalized modularity.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        comm.struct : dict
            Result of a community detection algorithm.
        gamma : double
            Resolution parameter.
        omega : double
            Interlayer coupling weight.
          
        Returns
        -------
        double
        
        References
        __________
        Mucha, Peter J., Richardson, Thomas, Macon, Kevin, Porter, Mason A., and Onnela, Jukka-Pekka (2010). Community structure in time-dependent, multiscale, and multiplex networks. Science (New York, N.Y.), 328(5980), 876-8. Data Analysis, Statistics and Probability; Physics and Society.
        
        See Also
        ________
        nmi
        omega_index
        )pbdoc");
    
    /*********************************************************************************/
    m.def("nmi", &nmi,
        py::arg("n"),
        py::arg("comm.struct1"),
        py::arg("comm.struct2"),
        R"pbdoc(
        Normalized mutual information.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        comm.struct1 : dict
            Result of a community detection algorithm.
        comm.struct2 : dict
            Result of a community detection algorithm.
        
        Returns
        -------
        double
        
        See Also
        ________
        modularity
        omega_index
        )pbdoc");
    
    
    /*********************************************************************************/
    m.def("omega_index", &omega,
        py::arg("n"),
        py::arg("comm.struct1"),
        py::arg("comm.struct2"),
        R"pbdoc(
        Omega index.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        comm.struct1 : dict
            Result of a community detection algorithm.
        comm.struct2 : dict
            Result of a community detection algorithm.
        
        Returns
        -------
        double
        
        See Also
        ________
        nmi
        omega_index
        )pbdoc");
    
    /**************************************/
    /* VISUALIZATION                  */
    /**************************************/
    
    /*********************************************************************************/
    m.def("layout_multiforce", &multiforce_ml,
        py::arg("n"),
        py::arg("w_in") = std::vector<double>({1}),
        py::arg("w_inter") = std::vector<double>({1}),
        py::arg("gravity") = std::vector<double>({0}),
        py::arg("iterations") = 100,
        R"pbdoc(
        Multiforce method: computes vertex coordinates.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        w_in : list of double
            An array with weights for intralayer forces, or a single number if weights are the same
            for all layers. When \code{w_in} is positive, vertices in the corresponding layer will
            be positioned as if a force was applied to them, repelling vertices that are close to
            each other and attracting adjacent vertices, all proportional to the specified weight.
        w_inter : list of double
            An array with weights for interlayer forces, or a single number if weights are the same
            for all layers. When \code{w_inter} is positive, vertices in the corresponding layer
            will be positioned as if a force was applied to them, trying to keep them aligned with
            the vertices corresponding to the same actors on other layers, proportionally to the
            specified weight.}
        gravity : list of double
            An array with weights for gravity forces, or a single number if weights are the same
            for all layers. This parameter results in the application of a force to the vertices,
            directed toward the center of the plot. It can be useful when there there are multiple
            components, so that they do not drift away from each other because of the repulsion
            force applied to their vertices.
        iterations : int
            Number of iterations.
        
        Returns
        -------
        dict
          "x", "y", "z" coordinates for each vertex ("actor", "layer").
        
        References
        __________
        Fatemi, Zahra, Salehi, Mostafa, & Magnani, Matteo (2018). A generalised force-based layout for multiplex sociograms. Social Informatics
        
        See Also
        ________
        layout_circular
        )pbdoc");
    
    /*********************************************************************************/
    m.def("layout_circular", &circular_ml,
        py::arg("n") ,
        R"pbdoc(
        Circular method: computes vertex coordinates arranging actors on a circle.
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        
        Returns
        -------
        dict
          "x", "y", "z" coordinates for each vertex ("actor", "layer").
        
        See Also
        ________
        layout_multiforce
        )pbdoc");
    
        // plotting function defined in functions.R
    
    
    /*********************************************************************************/
    /*m.def("get_community_list", &to_list,
        py::arg("comm.struct"),
        py::arg("n"),
        R"pbdoc(Converts a community structure (data frame) into a list of communities, layer by layer)pbdoc");
     */
    
    
    /**************************************/
    /* PYTHON-SPECIFIC                */
    /**************************************/

    /*********************************************************************************/
    m.def("to_edge_dict", &toNetworkxEdgeDict,
        py::arg("n") ,
        R"pbdoc(
        Returns a representation of the edges on each layer compatible with networkx.
        
        This utility function is internally used by to_nx_dict().
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        
        Returns
        -------
        dict
          In the format used by networkx to create graphs from dictionaries.
        
        See Also
        ________
        to_node_dict
        to_nx_dict
        )pbdoc");
    
    /*********************************************************************************/
    m.def("to_node_dict", &toNetworkxNodeDict,
        py::arg("n") ,
        R"pbdoc(
        Returns a representation of the actors on each layer compatible with networkx.
        
        This utility function is internally used by to_nx_dict().
        
        Parameters
        ----------
        n : PyMLNetwork
            A multilayer network.
        
        Returns
        -------
        dict
          In the format used by networkx to set node attributes.
        
        See Also
        ________
        to_edge_dict
        to_nx_dict
        )pbdoc");
    
    
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
