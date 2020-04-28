"""Multinet: Multilayer social network analysis and mining
    
This module defines a class to store multilayer networks and functions to pre-process, analyze
and mine them.

With multilayer social network we indicate a network where vertices (V) are organized into
multiple layers (L) and each node corresponds to an actor (A), where the same actor can be
mapped to nodes in different layers. Formally, a multilayer social network as implemented in
this package is a graph G = (V, E) where V is a subset of A x L.

Networks can be obtained from file (read) and written to file (write), some existing multilayer
networks are also included (data), synthetic multilayer networks can be generated (grow) and
layers can be added from networkx graphs (add_nx_layer).

Updating and getting information about the basic objects of a multilayer network can be done
using the functions add_obj, obj, delete_obj (where obj can be: layers, actors, vertices and
edges). neighbors retrieves the neighbors of a node. Attribute values can also be attached to
the basic objects in a multilayer network (actors, layers, vertices and edges) using the
functions attributes, add_attributes, get_values, set_values.

Each individual layer as well as combination of layers obtained using the data pre-processing
(flattening) functions can be analyzed as a single-layer network using the networkx package
(to_nx_dict). We can also visualize small networks using plot.

Multilayer network analysis measures are available for single-actors (degree, neighborhood,
xneighborhood, degree_deviation, relevance, xrelevance), based on geodesic distances (distance)
and to compare different layers (layer_summary, layer_comparison).

Communities can be extracted using various clustering algorithms: abacus, clique_percolation,
glouvain, infomap.

Most of the methods provided by this module are described in the book "Multilayer Social
Networks" [1]. These methods have been proposed by many different authors: extensive references
are available in the book, and in the documentation of each function we indicate the main
reference we have followed for the implementation. For a few methods developed after the book
was published we give specific references to the corresponding literature.
Additional information (including references to more learning material) is available in [2].

References
__________
.. [1] Dickison, Magnani, and Rossi, 2016. Multilayer Social Networks.
       Cambridge University Press. ISBN: 978-1107438750
.. [2] http://multilayer.it.uu.se/
"""

import networkx as nx
import pkgutil
import os


try:
    from uunet._multinet import (
    # creation
    empty,
    grow, evolution_pa, evolution_er,
    # io
    read, write,
    # info
    layers, actors, vertices, edges, edges_idx,
    num_layers, num_actors, num_vertices, num_edges,
    is_directed,
    # navigation
    neighbors, xneighbors,
    # manipulation
    add_layers, add_actors, add_vertices, add_edges,
    set_directed,
    delete_layers, delete_actors, delete_vertices, delete_edges,
    # attributes
    add_attributes, attributes, get_values, set_values,
    # transformation
    flatten,
    # measures
    degree, degree_deviation, neighborhood, xneighborhood, connective_redundancy,
    relevance, xrelevance,
    layer_summary, layer_comparison,
    distance,
    # clustering
    clique_percolation, glouvain, abacus, infomap,
    modularity,
    # visualization
    layout_multiforce, layout_circular,
    # networkx
    to_node_dict, to_edge_dict
    )
except ImportError as e:
    print('Failed importing!')
    print(e)


def data(name):
    """Loads one of the predefined datasets provided with the library.
        
    Parameters
    ----------
    name : str
        Can take the following values:
        - aucs: The AUCS multiplex network described in [1].
        - bankwiring: The Bankwiring network described in [2].
        - florentine: Padgett's Florentine Families multiplex network described in [3].
        - monastery: The Monastery network described in [2].
        - tailorshop: The Tailorshop network described in [4].
    
    Returns
    -------
    PyMLNetwork
        a multilayer network
        
    See Also
    ________
    read : from file
    grow : synthetic networks
    
    References
    __________
    .. [1] Rossi and Magnani (2015). "Towards effective visual analytics on multiplex networks". Chaos, Solitons and Fractals. Elsevier.
    .. [2] Breiger, R. and Boorman, S. and Arabic, P. (1975). An algorithm for clustering relational data with applications to social network analysis and comparison with multidimensional scaling. Journal of Mathematical Psychology, 12.
    .. [3] Padgett, John F., and McLean, Paul D. (2006). Organizational Invention and Elite Transformation: The Birth of Partnership Systems in Renaissance Florence. American Journal of Sociology, 111(5), 1463-1568.
    .. [4] Kapferer, Bruce (1972). Strategy and Transaction in an African Factory: African Workers and Indian Management in a Zambian Town. Manchester University Press.
    """
    if name == "aucs":
        path = os.path.dirname(globals()["__file__"])
        return read(path + "/data/aucs.mpx", name = "aucs")
    elif name == "florentine":
        path = os.path.dirname(globals()["__file__"])
        return read(path + "/data/florentine.mpx", name = "florentine")
    elif name == "bankwiring":
        path = os.path.dirname(globals()["__file__"])
        return read(path + "/data/bankwiring.mpx", name = "bankwiring")
    elif name == "monastery":
        path = os.path.dirname(globals()["__file__"])
        return read(path + "/data/monastery.mpx", name = "monastery")
    elif name == "tailorshop":
        path = os.path.dirname(globals()["__file__"])
        return read(path + "/data/tailorshop.mpx", name = "tailorshop")
    else:
        raise Exception("No predefined dataset named " + name)


def to_nx_dict(n):
    """Converts the network into one networkx object per layer.
        
    Parameters
    ----------
    n : PyMLNetwork
        A multilayer network.
    
    Returns
    -------
    dict
        A dictionary where each key is a layer and each value is a networkx graph (or directed graph)
        coresponding to that layer.
    """
    res = {}
    edges = to_edge_dict(n)
    nodes = to_node_dict(n)
    for layer in edges:
        if is_directed(n, [layer], [layer])["dir"][0]:
            res[layer] = nx.DiGraph(edges[layer])
        else:
            res[layer] = nx.Graph(edges[layer])
        nx.set_node_attributes(res[layer], nodes[layer])
    return res


def summary(n):
    """Produces basic layer-by-layer statistics.
        
    Parameters
    ----------
    n : PyMLNetwork
        A multilayer network.
        
    Returns
    -------
    dict
        A dictionary (representing a table) with the following columns:
        - "layer": name of the layer
        - "n": order
        - "m": size
        - "dir": directed/undirected (bool)
        - "nc": number of (strongly connected) components
        - "slc": size of the largest (strongly connected) component
        - "dens": density
        - "cc" : clustering coefficient (triangle-based definition)
        - "apl": average path length in the largest component
        - "dia": diameter
    """
    layers = to_nx_dict(n)
    res = {
        "layer": [],
        "n": [],
        "m": [],
        "dir": [],
        "nc": [],
        "slc": [],
        "dens": [],
        "cc" : [],
        "apl": [],
        "dia": [] }
    for l, net in layers.items():
        res["layer"].append(l)
        res["n"].append(net.order())
        res["m"].append(net.size())
        dir = nx.is_directed(net)
        res["dir"].append(dir)
        if dir:
            components = [net.subgraph(c) for c in nx.strongly_connected_components(net)]
            largest_component = max(components, key=len)
            res["nc"].append(nx.number_strongly_connected_components(net))
            res["slc"].append(len(largest_component))
            res["apl"].append(nx.average_shortest_path_length(largest_component))
            res["dia"].append(nx.diameter(largest_component))
        else:
            components = [net.subgraph(c) for c in nx.connected_components(net)]
            largest_component = max(components, key=len)
            res["nc"].append(nx.number_connected_components(net))
            res["slc"].append(len(largest_component))
            res["apl"].append(nx.average_shortest_path_length(largest_component))
            res["dia"].append(nx.diameter(largest_component))
        res["dens"].append(nx.density(net))
        res["cc"].append(nx.transitivity(net))
    return res


def add_nx_layer(n, g, name, node_attr = dict(), edge_attr = dict()):
    add_layers(n, [name], [nx.is_directed(g)])
    vertices = {"actor": g.nodes(), "layer": [name] * g.order() }
    add_vertices(n, vertices)
    for attr, type in node_attr:
        if type == "numeric":
            add_attributes_ml(n, attributes=[attr], type="numeric", target="vertex", layer=name)
            values = [el[1] for el in g.nodes(data=attr, default=0)]
            set_values_ml(n, attr, vertices=vertices, values=vertex_attr(g)[[attr]])
        elif type == "string":
            add_attributes_ml(n, attributes=[attr], type="string", target="vertex", layer=name)
            values = [el[1] for el in g.nodes(data=attr, default="")]
        # else exception
    edges = {
        "from_actor": [e[0] for e in g.edges()],
        "from_layer": [name] * g.size(),
        "to_actor": [e[1] for e in g.edges()],
        "to_layer": [name] * g.size()
    }
    add_edges(n, edges)

