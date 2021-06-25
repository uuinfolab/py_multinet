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
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import itertools
import functools

try:
    from uunet._multinet import (
    # creation
    empty,
    grow, evolution_pa, evolution_er,
    generate_communities,
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
    flatten, project,
    # measures
    degree, degree_deviation, neighborhood, xneighborhood, connective_redundancy,
    relevance, xrelevance,
    layer_summary, layer_comparison,
    distance,
    # clustering
    clique_percolation, glouvain, abacus, infomap, flat_ec, flat_nw, mdlp,
    modularity, nmi, omega_index,
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
        A dictionary representing a table with the following keys/columns:
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


def add_nx_layer(n, g, name, node_attr = {"name":[], "type":[]}, edge_attr = {"name":[], "type":[]}):
    """Adds a new layer to a network starting from a networkx graph.
    
    Parameters
    ----------
    n : PyMLNetwork
        A multilayer network.
    g :
        A networkx Graph or Digraph object.
    name : str
        Name of the new layer.
    node_attr : dict, optional
        Specifies which node attributes to copy to the layer. Must contain two lists:
        "name" (attribute names) and "type" (with values "string" or "numeric").
    edge_attr : dict, optional
        Specifies which edge attributes to copy to the layer. Must contain two lists:
        "name" (attribute names) and "type" (with values "string" or "numeric").
        
    See Also
    ________
    to_nx_dict : to turn layers into networkx graphs
    """
    add_layers(n, [name], [nx.is_directed(g)])
    vertices = {"actor": g.nodes(), "layer": [name] * g.order() }
    add_vertices(n, vertices)
    # @todo test & add eges
    for attr, type in zip(node_attr["name"], node_attr["type"]):
        if type == "numeric":
            add_attributes(n, attributes=[attr], type="numeric", target="vertex", layer=name)
            values = [el[1] for el in g.nodes(data=attr, default=0)]
            set_values_ml(n, attr, vertices=vertices, values=vertex_attr(g)[[attr]])
        elif type == "string":
            add_attributes(n, attributes=[attr], type="string", target="vertex", layer=name)
            values = [el[1] for el in g.nodes(data=attr, default="")]
        else: print("[Warning] wrong attribute type")
    edges = {
        "from_actor": [e[0] for e in g.edges()],
        "from_layer": [name] * g.size(),
        "to_actor": [e[1] for e in g.edges()],
        "to_layer": [name] * g.size()
    }
    add_edges(n, edges)



def values2graphics(values, output = ["c"], shape_size=7):
    """A utility function to turn nominal values into graphical features to be used by the plot function.
        
    Parameters
    ----------
    values : list
        A list of nominal values.
    output : list, optional
        A list of graphical features to be produced. Currently accepts "c" (colors) and "s" (shapes).
    shape_size : int, optional
        Size of the shapes associated to the values.
    
    Returns
    -------
    dict
        A dictionary with the following items:
        - "colors": a list of the same length of the input (values), where values have been
            replaced by colors.
        - "shapes": a list of the same length of the input (values), where values have been
            replaced by strings indicating shapes.
        - "legend": a list with one element for each distinct value in the input (values), each
            corresponding to an entry for the legend of the plotting function. Can be directly
            passed to the "legend" argument of plot.
    
    See Also
    ________
    plot : to plot multilayer networks
    """
    res = dict()
    types = list(set(values))
    num_types = len(types)
    if (num_types > 12):
        raise RuntimeError("only 12 distinct values supported")
    colors = ['black']*12
    if "c" in output:
        colors = plt.get_cmap("Paired").colors
    markers = ["s"]*12
    if "s" in output:
        markers = ["o","s","v","^","D","p","X","*","P",">","<","H"]
    res["legend"] = []
    for i in range(num_types):
        h = mlines.Line2D([], [], color=colors[i], marker=markers[i], linestyle='None', markersize=shape_size, label=types[i])
        res["legend"].append(h)
    res["colors"] = [colors[types.index(v)] for v in values]
    res["shapes"] = [markers[types.index(v)] for v in values]
    return res


def plot(n, base_size=3, mai=[.15,.15,.15,.15], layout=None, com=None, com_border=.4,
           #
           grid=[],
           #
           vertex_shape=['o'], vertex_size=[7], vertex_color=["black"], vertex_alpha=[1],
           #
           edge_style=["-"], edge_width=[1], edge_color=["black"], edge_alpha=[.8],
           edge_arrow_size=[30],
           #
           vertex_labels=None, vertex_labels_family="serif",
           vertex_labels_size=12, vertex_labels_weight="normal",
           vertex_labels_color="black",
           vertex_labels_style="normal", vertex_labels_stretch="normal",
           vertex_labels_bbox={"visible": False},
           vertex_labels_ha="center", vertex_labels_va="center",
           #
           legend=None, legend_loc="best", legend_ncol=1,
           show_layer_names=True, layer_names_bgcolor='white', layer_names_size=12,
           format="screen", file="mlfig"):
    """A (rudimentary) plotting function.
        
    This function uses matplotlib.pyplot drawing primitives. We repeat here some of
    the documentation; additional details can be found in pyplot's documentation.
    
    Parameters
    ----------
    n : PyMLNetwork
        A multilayer network.
    base_size : double, optional
        Approximate width and height of the part of the plot corresponding to one layer.
        In general, layers contain multiple layers organized in a grid (rows and columns), and the
        width (resp., height) of the plot will be base_size*num_cols (resp., *num_rows) plus margins
        and padding.
    mai : list of double, optional
        A list with four elements indicating the margin (left, top, right, bottom) inside each
        layer plot. The margin is applied to the vertex layout, that is, other graphical objects
        such as vertex labels can appear on top of the margin area. Margins are typically intended
        to make space for these additional graphical objects.
    layout : dict
        A dictionary representing a table with the following keys/columns:
        - "actor": name of the actor.
        - "layer": name of the layer.
        - "x": x coordinate of the vertex relative to the layer's plot.
        - "y": y coordinate of the vertex relative to the layer's plot.
        - "z": order of a layer (0, 1, ...).
    com : dict
        The result of a community detection algorithm. When provided, areas behind each community
        are represented in the plot.
    com_border : float
        Indicates the border around the community areas. If 0, the area passes through the center
        of the vertices in the convex hull of the vertices in the community, on each layer. A
        positive value increases the area.
    grid: list of double, optional
        A list with two elements, indicating number of rows and number of columns.
    vertex_shape : list of char, optional
        Characters indicating the shape of the vertices. It can contain one element, determining
        the shape of all vertices, or as many elements as the number of vertices.
        Any value allowed for matplotlib.markers is also allowed here (vertices are plotted using
        markers). Examples are 'o' (circle), 's' (square), 'D', 'd' (diamonds), 'v', '^', '<', '>'
        (triangles).
    vertex_size : list of int, optional
        Values indicating the size of the vertices. It can contain one element, determining
        the size of all vertices, or as many elements as the number of vertices.
    vertex_color : list of str, optional
        Values indicating the color of the vertices. It can contain one element, determining
        the color of all vertices, or as many elements as the number of vertices.
    vertex_alpha : list of double, optional
        Values indicating the transparency of the vertices, from 0 (transparent) to 1 (opaque).
        It can contain one element, determining the transparency of all vertices, or as many
        elements as the number of vertices.
    edge_style : list of str, optional
        Values indicating the line style of the edges/arcs.
        It can contain one element, determining the style of all edges, or as many
        elements as the number of edges. Examples of valid styles are: '-', '--', '-.', ':', '',
        (offset, on-off-seq).
    edge_width : list of int, optional
        Values indicating the line width of the edges/arcs.
        It can contain one element, determining the width of all edges, or as many
        elements as the number of edges.
    edge_color : list of str, optional
        Values indicating the color of the edges/arcs.
        It can contain one element, determining the color of all edges, or as many
        elements as the number of edges.
    edge_alpha : list of double, optional
        Values indicating the transparency of the edges/arcs, from 0 (transparent) to 1 (opaque).
        It can contain one element, determining the transparency of all edges, or as many
        elements as the number of edges.
    edge_arrow_size : list of int, optional
        Only for directed edges (arcs), values indicating the size of the arrow of the arcs.
        It can contain one element, determining the arrow size of all arcs, or as many
        elements as the number of arcs.
    vertex_labels : list of str, optional
        Values indicating the labels of the vertices. It must contain as many elements as the
        number of vertices, or a single value which will be applied to all vertices (in this case
        typically being [""], not to print any labels). If this parameter is not specified,
        actor names are visualized.
    vertex_labels_family : str, optional
        One of 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace', or a supported fontname.
    vertex_labels_size : int, optional
        Font size.
    vertex_labels_weight :
        This can be a numeric value in range 0-1000, 'ultralight', 'light', 'normal', 'regular',
        'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold',
        or 'black'.
    vertex_labels_color : str, optional
        A string representing a python color ('black', 'red', ...).
    vertex_labels_style : str, optional
        'normal', 'italic' or 'oblique'.
    vertex_labels_stretch : str, optional
        This can be a numeric value in range 0-1000, 'ultra-condensed', 'extra-condensed',
        'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded',
        or 'ultra-expanded'.
    vertex_labels_bbox : dict, optional
        A dictionary with properties as in the documentation for patches.FancyBboxPatch.
    vertex_labels_ha : str, optional
        Can be 'center', 'right', or 'left'.
    vertex_labels_va : str, optional
        Can be 'center', 'top', 'bottom', 'baseline', or 'center_baseline'.
    legend : list of Artist, optional
        This parameter is equivalent to the handles parameter of pyplot.legend(). It can be easier
        to set it using the values2graphics function, that automatically produces the value for this
        parameter for nominal values.
    legend_loc : str, optional
        If a legend is provided (see legend parameter), this parameter controls its location. Can
        be 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left',
        'center right', 'lower center', 'upper center', 'center', or a 2-tuple giving the
        coordinates of the lower-left corner of the legend in axes coordinates (from 0 to 1).
    legend_ncol : int, optional
        Number of columns in the legend.
    show_layer_names : bool, optional
        If True, draws the name of the layer below each layer plot.
    layer_names_bgcolor : str, optional
        If show_layer_names is true, this parameter controls the background color
        of the layer names.
    layer_names_size : int, optional
        If show_layer_names is true, this parameter controls the size of the layer names.
    format : str, optional
        This values specifies whether the plot should be shown in a new window ("screen", default)
        or saved to a file, in which case it specifies the image format ("pdf", "png", ...).
    file : str, optional
        If the format parameter indicates that the plot should be saved to file, this parameter
        allows to choose a file name.
    
    See Also
    ________
    values2graphics : to code nominal values into colors or shapes, also for the legend
    matplotlib.markers : for vertex shapes
    """
    ######################################
    # Some sub-functions used by plot(). #
    ######################################
    # Functions to compute planar coordinates (plot shows the layers beside each other, not stacked)
    def x_coord(x, y, z):
        return x + z%num_cols*width
    def y_coord(x, y, z):
        return y + (num_rows - 1 - z//num_cols)*height
    # Functions to draw an area behind groups of vertices, used to visualize communities.
    # Groups vertices from the result of a community detection algorithms into communities.
    def to_community_list(com):
        res = {}
        for a,l,cid in zip(com["actor"],com["layer"],com["cid"]):
            if cid not in res.keys():
                res[cid] = dict()
            if l not in res[cid].keys():
                res[cid][l] = []
            res[cid][l].append(a)
        return res
    # Produces a dictionary for quick retrieval of vertex coord., from the result of a layout algo.
    def index_layout(layout):
        res = {}
        for a,l,x,y,z in zip(layout["actor"], layout["layer"], layout["x"], layout["y"], layout["z"]):
            if a not in res.keys():
                res[a] = dict()
            res[a][l] = [x,y,z]
        return res
    # The following set of functions compute a ConvexHull (I think). The main function to do so
    # is perimeter().
    # (The ConvexHull function provided by scipy crashes on some systems.)
    def skyline(points, i, j):
        def comp(p1, p2, i, j):
            if i*p1[0]>i*p2[0]:
                if j*p1[1]>=j*p2[1]:
                    return 1
                else:
                    return 0
            if i*p1[0]<i*p2[0]:
                if j*p1[1]<=j*p2[1]:
                    return -1
                else:
                    return 0
            if j*p1[1]>j*p2[1]:
                return 1
            elif j*p1[1]<j*p2[1]:
                return -1
            return 1
        res = set()
        for p in points:
            dominated = False
            to_remove = set()
            for q in res:
                comparison = comp(p,q,i,j)
                if comparison==1:
                    to_remove.add(q)
                if comparison==-1:
                    dominated = True
                    break
            if dominated==False:
                for q in to_remove:
                    res.remove(q)
                res.add(p)
        return res
    def scomp(p1,p2,i):
        if i*p1[0]>i*p2[0]:
            return 1
        return -1
    def convex(points):
        res = []
        p = 0
        while p < len(points):
            res.append(points[p])
            max_slope=None
            idx=p+1
            for q in range(p+1,len(points)):
                slope = (points[q][1]-points[p][1])/(points[q][0]-points[p][0])
                if max_slope is None:
                    max_slope=slope
                elif slope>=max_slope:
                    max_slope=slope
                    idx = q
            p = idx
        return res
    def sky_ne(points):
        return skyline(points,1,1)
    def sky_se(points):
        return skyline(points,1,-1)
    def sky_nw(points):
        return skyline(points,-1,1)
    def sky_sw(points):
        return skyline(points,-1,-1)
    def c_nw(p1,p2):
        return scomp(p1,p2,1)
    def c_ne(p1,p2):
        return scomp(p1,p2,1)
    def c_sw(p1,p2):
        return scomp(p1,p2,-1)
    def c_se(p1,p2):
        return scomp(p1,p2,-1)
    def perimeter(points):
        per_nw = sorted(sky_nw(points), key=functools.cmp_to_key(c_nw))
        per_ne = sorted(sky_ne(points), key=functools.cmp_to_key(c_ne))
        per_se = sorted(sky_se(points), key=functools.cmp_to_key(c_se))
        per_sw = sorted(sky_sw(points), key=functools.cmp_to_key(c_sw))
        return convex(per_nw) + convex(per_ne) + convex(per_se) + convex(per_sw)
    # Given the results of to_community_list and index_layout, it draws polygons containing the
    # vertices inside communities.
    def draw_groups(com_list,coord):
        colors = plt.get_cmap("Set1").colors
        if (len(com_list) > 12):
            print("[Warning] some colors reused for different communities")
        for cid, l_dict in com_list.items():
            for l, actors in l_dict.items():
                point_set = set()
                for a in actors:
                    xyz = coord[a][l]
                    x = x_coord(xyz[0],xyz[1],xyz[2])
                    y = y_coord(xyz[0],xyz[1],xyz[2])
                    point_set.add((x-com_border*3/4,y-com_border*3/4))
                    point_set.add((x+com_border*3/4,y-com_border*3/4))
                    point_set.add((x+com_border*3/4,y+com_border*3/4))
                    point_set.add((x-com_border*3/4,y+com_border*3/4))
                    point_set.add((x-com_border,y))
                    point_set.add((x+com_border,y))
                    point_set.add((x,y+com_border))
                    point_set.add((x,y-com_border))
                per = perimeter(point_set)
                x = [p[0] for p in per]
                y = [p[1] for p in per]
                plt.fill(x,y,color=colors[cid],alpha=.8)
    ######################################
    # Main code of plot startig here.    #
    ######################################
    # checking argument: mai
    if len(mai) != 4:
        raise RuntimeError("argument mai must contain four values")
    if max(mai) > 1:
        print("[Warning] too high values for argument mai?")
    # computing layout if needed
    if layout is None:
        layout = layout_multiforce(n)
    # checking argument: vertex_shape, etc.
    if (len(vertex_shape) != 1) & (len(vertex_shape) != len(layout["actor"])):
        raise RuntimeError("argument vertex_shape must contain one element, or num_vertices elements.")
    if (len(vertex_size) != 1) & (len(vertex_size) != len(layout["actor"])):
        raise RuntimeError("argument vertex_size must contain one element, or num_vertices elements.")
    if (len(vertex_color) != 1) & (len(vertex_color) != len(layout["actor"])):
        raise RuntimeError("argument vertex_color must contain one element, or num_vertices elements.")
    if (len(vertex_alpha) != 1) & (len(vertex_alpha) != len(layout["actor"])):
        raise RuntimeError("argument vertex_alpha must contain one element, or num_vertices elements.")
    # checking argument: edge_style, etc.
    if (len(edge_width) != 1) & (len(edge_width) != num_edges(n)):
        raise RuntimeError("argument edge_width must contain one element, or num_edges elements.")
    if (len(edge_style) != 1) & (len(edge_style) != num_edges(n)):
        raise RuntimeError("argument edge_style must contain one element, or num_edges elements.")
    if (len(edge_color) != 1) & (len(edge_color) != num_edges(n)):
        raise RuntimeError("argument edge_color must contain one element, or num_edges elements.")
    if (len(edge_alpha) != 1) & (len(edge_alpha) != num_edges(n)):
        raise RuntimeError("argument edge_alpha must contain one element, or num_edges elements.")
    # option to select and reorder the layers
    # @todo only in the R version
    # grid, num columns, computing extreme coordinates...
    n_layers = num_layers(n)
    num_cols = n_layers
    num_rows = 1
    if len(grid) > 0:
        if len(grid) != 2:
            raise RuntimeError("argument grid must have two elements")
        if grid[0]*grid[1] < n_layers:
            raise RuntimeError("insufficient number of grid cells (< n_layers)")
        num_cols = grid[1]
        num_rows = grid[0]
    x_min = min(layout["x"])
    y_min = min(layout["y"])
    x_max = max(layout["x"])
    y_max = max(layout["y"])
    width = x_max-x_min + (mai[0]+mai[2])*(x_max-x_min)
    x_min = x_min - mai[0]*(x_max-x_min)
    height = y_max-y_min + (mai[1]+mai[3])*(y_max-y_min)
    y_min = y_min - mai[3]*(y_max-y_min)
    # create a figure
    f = plt.figure(figsize=(base_size*num_cols, base_size*num_rows))
    # draw grid
    for i in range(num_cols+1):
        plt.plot([x_min+i*width, x_min+i*width], [y_min, y_min+height*num_rows], 'k-', lw=1)
    for i in range(num_rows+1):
        plt.plot([x_min, x_min+width*num_cols], [y_min+i*height, y_min+i*height], 'k-', lw=1)
    # draw communities
    if com is not None:
        draw_groups(to_community_list(com), index_layout(layout))
    # draw edges
    e = edges_idx(n)
    for (fr, to, dir, style, size, color, alpha, arrow_w) in itertools.zip_longest(e["from"], e["to"], e["dir"], edge_style, edge_width, edge_color, edge_alpha, edge_arrow_size):
        fr_x = layout["x"][fr-1]
        fr_y = layout["y"][fr-1]
        fr_z = layout["z"][fr-1]
        to_x = layout["x"][to-1]
        to_y = layout["y"][to-1]
        to_z = layout["z"][to-1]
        marker = dict()
        if style is not None:
            st = style
        else:
            st = edge_style[0]
        if size is not None:
            wi = size
        else:
            wi = edge_width[0]
        if color is not None:
            co = color
        else:
            co = edge_color[0]
        if alpha is not None:
            al = alpha
        else:
            al = edge_alpha[0]
        if arrow_w is not None:
            arw = arrow_w
        else:
            arw = edge_arrow_size[0]
        x1 = x_coord(fr_x, fr_y, fr_z)
        y1 = y_coord(fr_x, fr_y, fr_z)
        x2 = x_coord(to_x, to_y, to_z)
        y2 = y_coord(to_x, to_y, to_z)
        if dir:
            plt.annotate('',
                         xytext=(x1, y1),
                         xy=(x2, y2),
                         arrowprops=dict(arrowstyle="-|>", lw=wi, ls=st, color=co, alpha=al),
                         size=arw
                         )
        else:
            plt.plot([x1, x2], [y1, y2], lw=wi, ls=st, color=co, alpha=al)
    # draw vertices
    for (x, y, z, shape, size, color, alpha) in itertools.zip_longest(layout["x"], layout["y"], layout["z"], vertex_shape, vertex_size, vertex_color, vertex_alpha):
        marker = dict()
        if shape is not None:
            marker["marker"] = shape
        else:
            marker["marker"] = vertex_shape[0]
        if size is not None:
            marker["markersize"] = size
        else:
            marker["markersize"] = vertex_size[0]
        if color is not None:
            marker["color"] = color
        else:
            marker["color"] = vertex_color[0]
        if alpha is not None:
            marker["alpha"] = alpha
        else:
            marker["alpha"] = vertex_alpha[0]
        plt.plot(x_coord(x,y,z), y_coord(x,y,z), **marker)
    # draw labels
    if vertex_labels is None:
        for (v, x, y, z) in zip(layout["actor"], layout["x"], layout["y"], layout["z"]):
            plt.text(x_coord(x,y,z), y_coord(x,y,z), v, family=vertex_labels_family,
                     size=vertex_labels_size, weight=vertex_labels_weight,
                     color=vertex_labels_color,
                     style=vertex_labels_style, stretch=vertex_labels_stretch,
                     bbox=vertex_labels_bbox, ha=vertex_labels_ha, va=vertex_labels_va)
    else:
        for (v, x, y, z) in zip(vertex_labels, layout["x"], layout["y"], layout["z"]):
            plt.text(x_coord(x,y,z), y_coord(x,y,z), v, family=vertex_labels_family,
                     size=vertex_labels_size, weight=vertex_labels_weight,
                     color=vertex_labels_color,
                     style=vertex_labels_style, stretch=vertex_labels_stretch,
                     bbox=vertex_labels_bbox, ha=vertex_labels_ha, va=vertex_labels_va)
    # draw layer names
    if show_layer_names:
        layer_names = layers(n)
        for i in range(n_layers):
            row = i // num_cols
            col = i % num_cols
            plt.text(x_min+col*width+width/2, y_min+(num_rows-row-1)*height, layer_names[i],
                     ha="center", va="center", bbox={"visible":True, "color":layer_names_bgcolor},
                     size=layer_names_size)
    # legend
    if legend is not None:
        plt.legend(handles=legend, loc=legend_loc, ncol=legend_ncol)
    # show plot
    plt.axis(False)
    plt.tight_layout()
    if format == "screen":
        plt.show()
    else:
        plt.savefig(file, format=format)

