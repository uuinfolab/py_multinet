import networkx as nx
import pkgutil
import os


try:
    from uunet._multinet import (
    # creation
    empty,
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
    """Loads one of the predefined datasets provided with the library"""
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
    """A function to convert the network into one networkx object per layer"""
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
        "actor_from": [e[0] for e in g.edges()],
        "layer_from": [name] * g.size(),
        "actor_to": [e[1] for e in g.edges()],
        "layer_to": [name] * g.size()
    }
    add_edges(n, edges)

