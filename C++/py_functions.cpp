// @todo check dict field existence

#include <sstream>
#include <limits>
#include "py_functions.hpp"
#include "pycpp_utils.hpp"

#include "operations/union.hpp"
#include "operations/project.hpp"
#include "community/glouvain2.hpp"
#include "community/abacus.hpp"
#include "community/infomap.hpp"
#include "community/flat.hpp"
#include "community/mlp.hpp"
#include "community/mlcpm.hpp"
#include "community/modularity.hpp"
#include "community/nmi.hpp"
#include "community/omega_index.hpp"
#include "io/read_multilayer_network.hpp"
#include "io/write_multilayer_network.hpp"
#include "measures/degree_ml.hpp"
#include "measures/neighborhood.hpp"
#include "measures/relevance.hpp"
#include "measures/redundancy.hpp"
#include "measures/layer.hpp"
#include "measures/distance.hpp"
#include "generation/communities.hpp"
#include "generation/evolve.hpp"
#include "generation/PAModel.hpp"
#include "generation/ERModel.hpp"
#include "core/propertymatrix/summarization.hpp"
#include "layout/multiforce.hpp"
#include "layout/circular.hpp"

using M = uu::net::MultilayerNetwork;
using G = uu::net::Network;

    // CREATION AND STORAGE

PyMLNetwork
emptyMultilayer(
    const std::string& name
)
{
    return PyMLNetwork(std::make_shared<uu::net::MultilayerNetwork>(name));
}

PyMLNetwork
readMultilayer(const std::string& input_file,
               const std::string& name, bool vertex_aligned)
{
    return PyMLNetwork(uu::net::read_multilayer_network(input_file,name,vertex_aligned));
}


void
writeMultilayer(
    const PyMLNetwork& rmnet,
    const std::string& output_file,
    const std::string& format,
    const py::list& layer_names,
    char sep,
    bool merge_actors,
    bool all_actors
)
{
    auto mnet = rmnet.get_mlnet();
    auto layers = resolve_layers_unordered(mnet, layer_names);

    if (format=="multilayer")
    {
        write_multilayer_network(mnet,layers.begin(),layers.end(),output_file,sep);
    }

    else if (format=="graphml")
    {
        if (!merge_actors && all_actors)
        {
            py::print("option all.actors not used when merge.actors=FALSE");
        }

        write_graphml(mnet,layers.begin(),layers.end(),output_file,merge_actors,all_actors);
    }

    else
    {
        throw std::runtime_error("unexpected value: format " + format);
    }
}

PyEvolutionModel
ba_evolution_model(
    size_t m0,
    size_t m
)
{
    auto pa = std::make_shared<uu::net::PAModel<uu::net::MultilayerNetwork>>(m0, m);

    return PyEvolutionModel(pa,"Preferential attachment evolution model (" + std::to_string(m0) + "," + std::to_string(m) + ")");
}


PyEvolutionModel
er_evolution_model(
    size_t n
)
{
    auto er = std::make_shared<uu::net::ERModel<uu::net::MultilayerNetwork>>(n);

    return PyEvolutionModel(er, "Uniform evolution model (" + std::to_string(n) + ")");
}


PyMLNetwork
growMultiplex(
    size_t num_actors,
    long num_of_steps,
    const py::list& evolution_model,
    const py::list& pr_internal_event,
    const py::list& pr_external_event,
    const py::list& dependency
)
{

    if (num_actors<=0)
    {
        throw std::runtime_error("The number of actors must be positive");
    }

    if (num_of_steps<=0)
    {
        throw std::runtime_error("The number of steps must be positive");
    }

    size_t num_layers = evolution_model.size();

    
    if (dependency.size() == 0)
    {
        throw std::runtime_error("Empty dependency matrix");
    }
    
    if (dependency.size() != num_layers
        || pr_internal_event.size() != num_layers
        || pr_external_event.size() != num_layers)
    {
        throw std::runtime_error("The number of evolution models, evolution probabilities and the number of rows of the dependency matrix must be the same");
    }

    if (dependency.size() != dependency[0].cast<py::list>().size())
    {
        throw std::runtime_error("The number of rows/columns of the dependency matrix must be the same");
    }
    
    std::vector<double> pr_int(pr_internal_event.size());

    for (size_t i=0; i<pr_internal_event.size(); i++)
    {
        pr_int[i] = pr_internal_event[i].cast<double>();
    }

    std::vector<double> pr_ext(pr_external_event.size());

    for (size_t i=0; i<pr_external_event.size(); i++)
    {
        pr_ext[i] = pr_external_event[i].cast<double>();
    }

    std::vector<std::vector<double> > dep;

    for (size_t i=0; i<dependency.size(); i++)
    {
        dep.push_back(std::vector<double>());

        for (size_t j=0; j<dependency.size(); j++)
        {
            dep[i].push_back( dependency[i].cast<py::list>()[j].cast<double>() );
        }
    }

    std::vector<uu::net::EvolutionModel<uu::net::MultilayerNetwork>*> models(evolution_model.size());

    for (size_t i=0; i<models.size(); i++)
    {
        models[i] = evolution_model[i].cast<PyEvolutionModel>().get_model();

    }

    auto res = std::make_shared<uu::net::MultilayerNetwork>("synth");

    std::vector<std::string> layer_names;

    for (size_t l=0; l<num_layers; l++)
    {
        std::string layer_name = "l"+std::to_string(l);
        //auto layer = std::make_unique<uu::net::Network>(layer_name, uu::net::EdgeDir::UNDIRECTED, true);
        res->layers()->add(layer_name, uu::net::EdgeDir::UNDIRECTED, uu::net::LoopMode::ALLOWED);
        layer_names.push_back(layer_name);
    }
    
    uu::net::evolve(res.get(), num_actors, layer_names, pr_int, pr_ext, dep, models, num_of_steps);
    
    return PyMLNetwork(res);
}

py::dict
generateCommunities(
     const std::string& type,
     size_t num_actors,
     size_t num_layers,
     size_t num_communities,
     size_t overlap,
     const py::list& pr_internal,
     const py::list& pr_external
)
{
    // @todo wrap code to process vectors in a utility function
    std::vector<double> pr_int(num_layers);
    if (pr_internal.size() == 1)
    {
        for (size_t i=0; i<num_layers; i++)
        {
            pr_int[i] = pr_internal[0].cast<double>();
        }
    }
    else if (pr_internal.size() == num_layers)
    {
        for (size_t i=0; i<num_layers; i++)
        {
            pr_int[i] = pr_internal[i].cast<double>();
        }
    }
    else throw uu::core::WrongParameterException("wrong number of values in pr.internal");
    
    std::vector<double> pr_ext(num_layers);
    if (pr_external.size() == 1)
    {
        for (size_t i=0; i<num_layers; i++)
        {
            pr_ext[i] = pr_external[0].cast<double>();
        }
    }
    else if (pr_external.size() == num_layers)
    {
        for (size_t i=0; i<num_layers; i++)
        {
            pr_ext[i] = pr_external[i].cast<double>();
        }
    }
    else throw uu::core::WrongParameterException("wrong number of values in pr.external");
    
    std::string uc_type = type;
    uu::core::to_upper_case(uc_type);
    
    if (uc_type == "PEP")
    {
        if (overlap > 0)
        {
            py::print("[Warning] unused parameter: \"overlap\"");
        }
        auto pair = uu::net::generate_pep(num_layers, num_actors, num_communities, pr_int, pr_ext);
        py::dict res;
        res["net"]=PyMLNetwork(std::move(pair.first));
        res["com"]=to_dataframe(pair.second.get());
        return res;
    }
    else if (uc_type == "PEO")
    {
        auto pair = uu::net::generate_peo(num_layers, num_actors, num_communities, overlap, pr_int, pr_ext);
        py::dict res;
        res["net"]=PyMLNetwork(std::move(pair.first));
        res["com"]=to_dataframe(pair.second.get());
        return res;
    }
    else if (uc_type == "SEP")
    {
        if (overlap > 0)
        {
            py::print("[Warning] unused parameter: \"overlap\"");
        }
        auto pair = uu::net::generate_sep(num_layers, num_actors, num_communities, pr_int, pr_ext);
        py::dict res;
        res["net"]=PyMLNetwork(std::move(pair.first));
        res["com"]=to_dataframe(pair.second.get());
        return res;
    }
    else if (uc_type == "SEO")
    {
        auto pair = uu::net::generate_seo(num_layers, num_actors, num_communities, overlap, pr_int, pr_ext);
        py::dict res;
        res["net"]=PyMLNetwork(std::move(pair.first));
        res["com"]=to_dataframe(pair.second.get());
        return res;
    }
    else throw uu::core::WrongParameterException("wrong type parameter");
    return py::dict();
}


// INFORMATION ON NETWORKS

py::list
layers(
    const PyMLNetwork& rmnet
)
{
    auto mnet = rmnet.get_mlnet();
    py::list res;

    for (auto layer: *mnet->layers())
    {
        res.append(layer->name);
    }

    return res;
}

py::dict
actors(
    const PyMLNetwork& rmnet,
    const py::list& layer_names,
    bool add_attributes
)
{
    py::list actors;
    auto mnet = rmnet.get_mlnet();

    if (layer_names.size()==0)
    {
        for (auto actor: *mnet->actors())
        {
            actors.append(actor->name);
        }
    }

    else
    {
        auto layers = resolve_layers(mnet,layer_names);

        for (auto layer: layers)
        {
            for (auto vertex: *layer->vertices())
            {
                actors.append(vertex->name);
            }
        }
    }

    py::dict res;
    res["actor"] = actors;
    
    if (add_attributes)
    {
        auto attrs = mnet->actors()->attr();
        for (auto attr: *attrs)
        {
            if (attr->name == "actor")
            {
                throw std::runtime_error("attribute name \"actor\" already present in the dictionary");
            }
            auto values = getValues(rmnet, attr->name, res, py::dict(), py::dict());
            res[pybind11::cast(attr->name)] = values[pybind11::cast(attr->name)];
        }
    }
    return res;
}

py::dict
vertices(
    const PyMLNetwork& rmnet,
    const py::list& layer_names,
    bool add_attributes
)
{
    auto mnet = rmnet.get_mlnet();
    auto layers = resolve_layers_unordered(mnet,layer_names);
    py::list actor, layer;

    for (auto l: layers)
    {

        /*if (layers.count(l)==0)
        {
            continue;
        }*/

        for (auto vertex: *l->vertices())
        {
            actor.append(vertex->name);
            layer.append(l->name);
        }
    }

    py::dict res;
    res["actor"] = actor;
    res["layer"] = layer;
    
    if (add_attributes)
    {
        // Adding actor attributes
        auto attrs = mnet->actors()->attr();
        for (auto attr: *attrs)
        {
            if (attr->name == "actor")
            {
                throw std::runtime_error("attribute name \"actor\" already present in the dictionary");
            }
            if (attr->name == "layer")
            {
                throw std::runtime_error("attribute name \"layer\" already present in the dictionary");
            }
            auto values = getValues(rmnet, attr->name, res, py::dict(), py::dict());
            res[pybind11::cast(attr->name)] = values[pybind11::cast(attr->name)];
        }
        // Adding vertex (that is, layer-specific) attributes
        for (auto l: layers)
        {
            auto attrs = l->vertices()->attr();
            for (auto attr: *attrs)
            {
                if (attr->name == "actor")
                {
                    throw std::runtime_error("attribute name \"actor\" already present in the dictionary");
                }
                if (attr->name == "layer")
                {
                    throw std::runtime_error("attribute name \"layer\" already present in the dictionary");
                }
                auto values = getValues(rmnet, attr->name, py::dict(), res, py::dict());
                for (auto item : values)
                {
                    res[item.first] = item.second;
                }
            }
        }
    }
    return res;
}

py::dict
edges_idx(
    const PyMLNetwork& rmnet
)
{
    auto mnet = rmnet.get_mlnet();
    py::list from, to, directed;

    // stores at which index vertices start in a layer
    std::unordered_map<const uu::net::VCube*, size_t> offset;
    size_t num_vertices = 0;
    for (auto layer: *mnet->layers())
    {
        offset[layer->vertices()] = num_vertices;
        num_vertices += layer->vertices()->size();
    }
    
    // intralayer
    
    for (auto l: *mnet->layers())
    {
        auto vertices = l->vertices();

        for (auto edge: *l->edges())
        {
            from.append(vertices->index_of(edge->v1)+offset[edge->c1]+1);
            to.append(vertices->index_of(edge->v2)+offset[edge->c2]+1);
            directed.append((edge->dir==uu::net::EdgeDir::DIRECTED)?1:0);
        }
    }

    // interlayer
    for (auto l1: *mnet->layers())
    {
        //num_edges += l1->edges()->size();
        for (auto l2: *mnet->layers())
        {
            if (l2 <= l1) continue;
            auto edges = mnet->interlayer_edges()->get(l1, l2);
            if (!edges) continue;
            for (auto edge: *edges)
            {
                from.append(edge->c1->index_of(edge->v1)+offset[edge->c1]+1);
                to.append(edge->c2->index_of(edge->v2)+offset[edge->c2]+1);
                directed.append((edge->dir==uu::net::EdgeDir::DIRECTED)?1:0);
            }
        }
    }

    py::dict res;
    res["from"] = from;
    res["to"] = to;
    res["dir"] = directed;
    return res;
}

py::dict
edges(
    const PyMLNetwork& rmnet,
    const py::list& layer_names1,
    const py::list& layer_names2,
    bool add_attributes
)
{
    auto mnet = rmnet.get_mlnet();
    std::vector<uu::net::Network*> layers1 = resolve_layers(mnet,layer_names1);
    std::vector<uu::net::Network*> layers2;

    if (layer_names2.size()==0)
    {
        layers2 = layers1;
    }

    else
    {
        layers2 = resolve_layers(mnet,layer_names2);
    }

    py::list from_a, from_l, to_a, to_l;
    py::list directed;

    for (auto layer1: layers1)
    {
        for (auto layer2: layers2)
        {
            if (layer2<layer1)
            {
                continue;
            }

            else if (layer1==layer2)
            {

                for (auto edge: *layer1->edges())
                {
                    from_a.append(edge->v1->name);
                    from_l.append(layer1->name);
                    to_a.append(edge->v2->name);
                    to_l.append(layer1->name);
                    directed.append((edge->dir==uu::net::EdgeDir::DIRECTED)?true:false);
                }
            }

            else
            {
                auto edges = mnet->interlayer_edges()->get(layer1,layer2);
                if (!edges) continue;
                for (auto edge: *edges)
                {
                    from_a.append(edge->v1->name);
                    from_l.append(edge->c1->name);
                    to_a.append(edge->v2->name);
                    to_l.append(edge->c2->name);
                    directed.append((edge->dir==uu::net::EdgeDir::DIRECTED)?true:false);
                }
            }

        }
    }
    py::dict res;
    res["from_actor"] = from_a;
    res["from_layer"] = from_l;
    res["to_actor"] = to_a;
    res["to_layer"] = to_l;
    res["dir"] = directed;
    
    if (add_attributes)
    {
        auto attributes = getAttributes(rmnet, "edge");
        std::set<std::string> attrs;
        for (auto a: attributes["name"])
        {
            attrs.insert(std::string(py::str(a)));
        }
        for (auto attr: attrs)
        {
            auto values = getValues(rmnet, attr, py::dict(), py::dict(), res);
            res[pybind11::cast(attr)] = values[pybind11::cast(attr)];
        }
    }
    
    return res;
}

size_t
numLayers(
    const PyMLNetwork& rmnet
)
{
    auto mnet = rmnet.get_mlnet();
    return mnet->layers()->size();
}

size_t
numActors(
    const PyMLNetwork& rmnet,
    const py::list& layer_names
)
{
    auto mnet = rmnet.get_mlnet();

    if (layer_names.size()==0)
    {
        return mnet->actors()->size();
    }

    std::vector<uu::net::Network*> layers = resolve_layers(mnet,layer_names);
    std::unordered_set<const uu::net::Vertex*> actors;

    for (auto layer: layers)
    {
        for (auto vertex: *layer->vertices())
        {
            actors.insert(vertex);
        }
    }

    return actors.size();
}

size_t
numNodes(
    const PyMLNetwork& rmnet,
    const py::list& layer_names
)
{
    auto mnet = rmnet.get_mlnet();
    std::vector<uu::net::Network*> layers = resolve_layers(mnet,layer_names);
    size_t num_vertices = 0;

    for (auto layer: layers)
    {
        num_vertices += layer->vertices()->size();
    }

    return num_vertices;
}

size_t
numEdges(
    const PyMLNetwork& rmnet,
    const py::list& layer_names1,
    const py::list& layer_names2
)
{
    auto mnet = rmnet.get_mlnet();
    std::unordered_set<const uu::net::Network*> layers1 = resolve_const_layers_unordered(mnet,layer_names1);
    std::unordered_set<const uu::net::Network*> layers2;

    if (layer_names2.size()==0)
    {
        layers2 = layers1;
    }

    else
    {
        layers2 = resolve_const_layers_unordered(mnet,layer_names2);
    }

    size_t num_edges = 0;



    for (auto layer1: layers1)
    {
        for (auto layer2: layers2)
        {
            if (layer2<layer1)
            {
                continue;
            }

            else if (layer1==layer2)
            {
                num_edges += layer1->edges()->size();
            }

            else
            {
                if (!mnet->interlayer_edges()->get(layer1,layer2))
                {
                    continue;
                }
                num_edges += mnet->interlayer_edges()->get(layer1,layer2)->size();
            }

        }
    }

    return num_edges;
}

py::dict
isDirected(
    const PyMLNetwork& rmnet,
    const py::list& layer_names1,
    const py::list& layer_names2)
{
    auto mnet = rmnet.get_mlnet();
    std::vector<uu::net::Network*> layers1 = resolve_layers(mnet,layer_names1);
    std::vector<uu::net::Network*> layers2;

    if (layer_names2.size()==0)
    {
        layers2 = layers1;
    }

    else
    {
        layers2 = resolve_layers(mnet,layer_names2);
    }

    py::list l1, l2;
    py::list directed;

    for (auto layer1: layers1)
    {
        for (auto layer2: layers2)
        {
            l1.append(layer1->name);
            l2.append(layer2->name);
            
            if (layer1==layer2)
            {
                directed.append(layer1->is_directed()?true:false);
            }
            else
            {
                if (!mnet->interlayer_edges()->get(layer1,layer2))
                {

                    py::print("[Warning] interlayer edges between " + layer1->name + " and " + layer2->name +
                    " not initialized");
                    continue;
                }
                directed.append((mnet->interlayer_edges()->is_directed(layer1,layer2))?true:false);
            }
        }
    }
    
    py::dict res;
    res["layer1"] = l1;
    res["layer2"] = l2;
    res["dir"] = directed;
    return res;
}

std::unordered_set<std::string>
actor_neighbors(
    const PyMLNetwork& rmnet,
    const std::string& actor_name,
    const py::list& layer_names,
    const std::string& mode_name
)
{
    std::unordered_set<std::string> res_neighbors;
    auto mnet = rmnet.get_mlnet();
    auto actor = mnet->actors()->get(actor_name);

    if (!actor)
    {
        throw std::runtime_error("actor " + actor_name + " not found");
    }

    auto layers = resolve_layers_unordered(mnet, layer_names);
    auto mode = resolve_mode(mode_name);
    auto actors = uu::net::neighbors(layers.begin(), layers.end(), actor, mode);

    for (auto neigh: actors)
    {
        res_neighbors.insert(neigh->name);
    }

    return res_neighbors;
}

std::unordered_set<std::string>
actor_xneighbors(
    const PyMLNetwork& rmnet,
    const std::string& actor_name,
    const py::list& layer_names,
    const std::string& mode_name
)
{
    std::unordered_set<std::string> res_xneighbors;
    auto mnet = rmnet.get_mlnet();
    auto actor = mnet->actors()->get(actor_name);

    if (!actor)
    {
        throw std::runtime_error("actor " + actor_name + " not found");
    }

    auto layers = resolve_layers_unordered(mnet,layer_names);
    auto mode = resolve_mode(mode_name);
    auto actors = uu::net::xneighbors(mnet, layers.begin(), layers.end(), actor, mode);

    for (auto neigh: actors)
    {
        res_xneighbors.insert(neigh->name);
    }

    return res_xneighbors;
}


// NETWORK MANIPULATION

void
addLayers(
    PyMLNetwork& rmnet,
    const py::list& layer_names,
    const py::list& directed
)
{
    auto mnet = rmnet.get_mlnet();

    auto layer_iter = layer_names.begin();
    auto dir_iter = directed.begin();
    
    if (directed.size()==0)
    {
        auto dir = uu::net::EdgeDir::UNDIRECTED;
        while (layer_iter != layer_names.end())
        {
            auto layer_name = (*layer_iter).cast<std::string>();
            //auto layer = std::make_unique<G>(layer_name, dir, true);
            mnet->layers()->add(layer_name, dir, uu::net::LoopMode::ALLOWED);
            ++layer_iter;
        }
    }

    else if (directed.size()==1)
    {
        bool directionality = (*dir_iter).cast<bool>();
        auto dir = (directionality)?uu::net::EdgeDir::DIRECTED:uu::net::EdgeDir::UNDIRECTED;
        while (layer_iter != layer_names.end())
        {
            auto layer_name = (*layer_iter).cast<std::string>();
            mnet->layers()->add(layer_name, dir, uu::net::LoopMode::ALLOWED);
            ++layer_iter;
        }
    }

    else if (layer_names.size()!=directed.size())
    {
        throw std::runtime_error("Same number of layer names and layer directionalities expected");
    }

    else
    {
        while (layer_iter != layer_names.end())
        {
            bool directionality = (*dir_iter).cast<bool>();
            auto dir = (directionality)?uu::net::EdgeDir::DIRECTED:uu::net::EdgeDir::UNDIRECTED;
            auto layer_name = (*layer_iter).cast<std::string>();
            //auto layer = std::make_unique<G>(layer_name, dir, true);
            mnet->layers()->add(layer_name, dir, uu::net::LoopMode::ALLOWED);
            ++layer_iter;
            ++dir_iter;
        }
    }
}

void
addActors(
    PyMLNetwork& rmnet,
    const py::list& actor_names
)
{
    auto mnet = rmnet.get_mlnet();

    for (py::handle obj: actor_names)
    {
        std::string actor_name = obj.attr("__str__")().cast<std::string>();
        mnet->actors()->add(actor_name);
    }
}

void
addNodes(
    PyMLNetwork& rmnet,
    const py::dict& vertices
    )
{
    auto mnet = rmnet.get_mlnet();

    py::list a = vertices["actor"].cast<py::list>();
    py::list l = vertices["layer"].cast<py::list>();

    /* From V4: actors are not added separately
    for (py::handle obj: a)
    {
        std::string actor_name = obj.attr("__str__")().cast<std::string>();
        mnet->actors()->add(actor_name);
    }
    */
    
    auto actor_iter = a.begin();
    auto layer_iter = l.begin();
    while (actor_iter != a.end())
    {
        std::string actor_name = (*actor_iter).attr("__str__")().cast<std::string>();
        std::string layer_name = (*layer_iter).attr("__str__")().cast<std::string>();
             
        auto layer = mnet->layers()->get(layer_name);

        if (!layer)
        {
            auto dir = uu::net::EdgeDir::UNDIRECTED;
            layer = mnet->layers()->add(layer_name, dir, uu::net::LoopMode::ALLOWED);
        }
        
        auto actor = mnet->actors()->get(actor_name);
        
        if (!actor)
        {
            layer->vertices()->add(actor_name);
        }
        else
        {
            layer->vertices()->add(actor);
        }
        
        
        
        ++actor_iter;
        ++layer_iter;
    }
}

void
addEdges(
    PyMLNetwork& rmnet,
    const py::dict& edges)
{
    auto mnet = rmnet.get_mlnet();

    py::list a_from = edges["from_actor"].cast<py::list>();
    py::list l_from = edges["from_layer"].cast<py::list>();
    py::list a_to = edges["to_actor"].cast<py::list>();
    py::list l_to = edges["to_layer"].cast<py::list>();

    
    /*
    for (py::handle obj: a_from)
    {
        std::string actor_name = obj.attr("__str__")().cast<std::string>();
        mnet->actors()->add(actor_name);
    }
    for (py::handle obj: a_to)
    {
        std::string actor_name = obj.attr("__str__")().cast<std::string>();
        mnet->actors()->add(actor_name);
    }
    */
    
    auto actor_from_iter = a_from.begin();
    auto layer_from_iter = l_from.begin();
    auto actor_to_iter = a_to.begin();
    auto layer_to_iter = l_to.begin();
    
    while (actor_from_iter != a_from.end())
    {
        std::string actor_name1 = (*actor_from_iter).attr("__str__")().cast<std::string>();
        std::string layer_name1 = (*layer_from_iter).attr("__str__")().cast<std::string>();
        std::string actor_name2 = (*actor_to_iter).attr("__str__")().cast<std::string>();
        std::string layer_name2 = (*layer_to_iter).attr("__str__")().cast<std::string>();
        
        auto layer1 = mnet->layers()->get(layer_name1);
        if (!layer1)
        {
            auto dir = uu::net::EdgeDir::UNDIRECTED;
            layer1 = mnet->layers()->add(layer_name1, dir, uu::net::LoopMode::ALLOWED);
        }
        
        auto actor1 = layer1->vertices()->get(actor_name1);
        if (!actor1)
        {
            actor1 = mnet->actors()->add(actor_name1);
        }

        auto layer2 = mnet->layers()->get(layer_name2);
        if (!layer2)
        {
            auto dir = uu::net::EdgeDir::UNDIRECTED;
            layer2 = mnet->layers()->add(layer_name2, dir, uu::net::LoopMode::ALLOWED);
        }

        auto actor2 = layer2->vertices()->get(actor_name2);
        if (!actor2)
        {
            actor2 = mnet->actors()->add(actor_name2);
        }
        
        if (layer1==layer2)
        {
            layer1->edges()->add(actor1, actor2);
        }

        else
        {
            if (!mnet->interlayer_edges()->get(layer1,layer2))
            {
                uu::net::EdgeDir dir = uu::net::EdgeDir::UNDIRECTED;
                mnet->interlayer_edges()->init(layer1,layer2, dir);
            }
            mnet->interlayer_edges()->add(actor1, layer1, actor2, layer2);
        }
        
        ++actor_from_iter;
        ++layer_from_iter;
        ++actor_to_iter;
        ++layer_to_iter;
    }
}

void
setDirected(
    const PyMLNetwork& rmnet,
    const py::dict& layers_dir)
{
    auto mnet = rmnet.get_mlnet();
    py::list l1 = layers_dir["layer1"].cast<py::list>();
    py::list l2 = layers_dir["layer2"].cast<py::list>();
    py::list dir = layers_dir["dir"].cast<py::list>();

    auto layer1_iter = l1.begin();
    auto layer2_iter = l2.begin();
    auto dir_iter = dir.begin();
    
    while (layer1_iter != l1.end())
    {
        std::string layer_name1 = (*layer1_iter).attr("__str__")().cast<std::string>();
        
        auto layer1 = mnet->layers()->get(layer_name1);

        if (!layer1)
        {
            throw std::runtime_error("cannot find layer " + layer_name1);
        }
        
        std::string layer_name2 = (*layer2_iter).attr("__str__")().cast<std::string>();
        
        auto layer2 = mnet->layers()->get(layer_name2);

        if (!layer2)
        {
            throw std::runtime_error("cannot find layer " + layer_name2);
        }

        bool directed = (*dir_iter).cast<bool>();

        if (layer1==layer2)
        {
            // @todo do nothing?
        }
        else
        {
            if (!mnet->interlayer_edges()->get(layer1,layer2))
            {
                uu::net::EdgeDir dir = directed?uu::net::EdgeDir::DIRECTED:uu::net::EdgeDir::UNDIRECTED;
                mnet->interlayer_edges()->init(layer1,layer2, dir);
            }
            else
            {
                py::print("[Warning] cannot initialize existing pair of layers " +
                layer1->name + " and " + layer2->name);
                continue;
            }
        }
        
        ++layer1_iter;
        ++layer2_iter;
        ++dir_iter;
        
    }
}

void
deleteLayers(
    PyMLNetwork& rmnet,
    const py::list& layer_names)
{
    auto mnet = rmnet.get_mlnet();

    for (py::handle obj: layer_names)
    {
        std::string layer_name = obj.attr("__str__")().cast<std::string>();
        auto layer = mnet->layers()->get(layer_name);
        mnet->layers()->erase(layer);
    }
}

void
deleteActors(
    PyMLNetwork& rmnet,
    const py::list& actor_names)
{
    auto mnet = rmnet.get_mlnet();

    for (py::handle obj: actor_names)
    {
        std::string actor_name = obj.attr("__str__")().cast<std::string>();
        auto actor = mnet->actors()->get(actor_name);
        if (actor) mnet->actors()->erase(actor);
    }
}

void
deleteNodes(
    PyMLNetwork& rmnet,
    const py::dict& vertex_matrix
)
{
    auto mnet = rmnet.get_mlnet();
    auto vertices = resolve_vertices(mnet, vertex_matrix);

    for (auto vertex: vertices)
    {
        auto actor = vertex.first;
        auto layer = vertex.second;
        layer->vertices()->erase(actor);
    }
}

void
deleteEdges(
    PyMLNetwork& rmnet,
    const py::dict& edge_matrix
)
{
    auto mnet = rmnet.get_mlnet();
    auto edges = resolve_edges(mnet, edge_matrix);

    for (auto edge: edges)
    {
        auto actor1 = std::get<0>(edge);
        auto layer1 = std::get<1>(edge);
        auto actor2 = std::get<2>(edge);
        auto layer2 = std::get<3>(edge);
        if (layer1 == layer2)
        {
            auto e = layer1->edges()->get(actor1, actor2);
            layer1->edges()->erase(e);
        }
        else
        {
            //auto e = mnet->interlayer_edges()->get(actor1, layer1, actor2, layer2);
            mnet->interlayer_edges()->erase(actor1, layer1, actor2, layer2);
        }
    }
}


void
newAttributes(
    PyMLNetwork& rmnet,
    const py::list& attribute_names,
    const std::string& type,
    const std::string& target,
    const std::string& layer_name,
    const std::string& layer_name1,
    const std::string& layer_name2
)
{
    auto mnet = rmnet.get_mlnet();

    uu::core::AttributeType a_type;

    if (type=="string")
    {
        a_type = uu::core::AttributeType::STRING;
    }

    else if (type=="numeric")
    {
        a_type = uu::core::AttributeType::DOUBLE;
    }

    else
    {
        throw std::runtime_error("Wrong type");
    }

    if (target=="actor")
    {
        if (layer_name!="" || layer_name1!="" || layer_name2!="")
        {
            throw std::runtime_error("No layers should be specified for target 'actor'");
        }

        for (py::handle obj: attribute_names)
        {
            std::string attr_name = obj.attr("__str__")().cast<std::string>();
            mnet->actors()->attr()->add(attr_name,a_type);
        }
    }

    else if (target=="layer")
    {
        throw std::runtime_error("layer attributes are not available in this version of the library");
    }

    else if (target=="node" || target=="vertex")
    {
        if (target=="node")
        {
            py::print("target 'node' deprecated: use 'vertex' instead");
        }

        if (layer_name1!="" || layer_name2!="")
        {
            throw std::runtime_error("layer1 and layer2 should not be specified for target '" + target + "'");
        }

        auto layer = mnet->layers()->get(layer_name);

        if (!layer)
        {
            throw std::runtime_error("layer " + layer_name + " not found");
        }

        for (py::handle obj: attribute_names)
        {
            std::string attr_name = obj.attr("__str__")().cast<std::string>();
            layer->vertices()->attr()->add(attr_name,a_type);
        }
    }

    else if (target=="edge")
    {
        if (layer_name!="" && (layer_name1!="" || layer_name2!=""))
        {
            throw std::runtime_error("either layers (for intra-layer edges) or layers1 and layers2 (for inter-layer edges) must be specified for target 'edge'");
        }

        uu::net::Network* layer1;
        uu::net::Network* layer2;

        if (layer_name1=="")
        {
            layer1 = mnet->layers()->get(layer_name);
            layer2 = layer1;

            if (!layer1)
            {
                throw std::runtime_error("layer " + layer_name + " not found");
            }
        }

        else if (layer_name2!="")
        {
            layer1 = mnet->layers()->get(layer_name1);
            layer2 = mnet->layers()->get(layer_name2);
        }

        else
        {
            throw std::runtime_error("if layer1 is specified, also layer2 is required");
        }

        if (layer1 == layer2)
        {
            for (py::handle obj: attribute_names)
            {
                std::string attr_name = obj.attr("__str__")().cast<std::string>();
                layer1->edges()->attr()->add(attr_name,a_type);
            }
        }

        else
        {
            throw std::runtime_error("attributes on inter-layer edges are not available in this version of the library");
        }

    }

    else
    {
        throw std::runtime_error("wrong target: " + target);
    }
}


py::dict
getAttributes(
    const PyMLNetwork& rmnet,
    const std::string& target
)
{
    auto mnet = rmnet.get_mlnet();

    if (target=="actor")
    {
        auto attributes = mnet->actors()->attr();
        py::list a_name, a_type;

        for (auto att: *attributes)
        {
            a_name.append(att->name);
            a_type.append(uu::core::to_string(att->type));
        }

        py::dict res;
        res["name"] = a_name;
        res["type"] = a_type;
        return res;
    }

    else if (target=="layer")
    {
        throw std::runtime_error("layer attributes are not available in this version of the library");
    }

    else if (target=="node" || target=="vertex")
    {
        if (target=="node")
        {
            py::print("target 'node' deprecated: use 'vertex' instead");
        }

        py::list a_layer, a_name, a_type;

        for (auto layer: *mnet->layers())
        {
            auto attributes = layer->vertices()->attr();

            for (auto att: *attributes)
            {
                a_layer.append(layer->name);
                a_name.append(att->name);
                a_type.append(uu::core::to_string(att->type));
            }
        }

        py::dict res;
        res["layer"] = a_layer;
        res["name"] = a_name;
        res["type"] = a_type;
        return res;
    }

    else if (target=="edge")
    {

        py::list a_layer, a_name, a_type;

        for (auto layer: *mnet->layers())
        {
            auto attributes = layer->edges()->attr();

            for (auto att: *attributes)
            {
                a_layer.append(layer->name);
                a_name.append(att->name);
                a_type.append(uu::core::to_string(att->type));
            }
        }

        auto attributes = mnet->interlayer_edges()->attr();
        
        for (auto att: *attributes)
        {
            a_layer.append("--");
            a_name.append(att->name);
            a_type.append(uu::core::to_string(att->type));
        }
        
        
        py::dict res;
        res["layer"] = a_layer;
        res["name"] = a_name;
        res["type"] = a_type;
        return res;
    }

    else
    {
        throw std::runtime_error("wrong target: " + target);
    }

    return py::dict(); // never gets here
}


py::dict
getValues(
    const PyMLNetwork& rmnet,
    const std::string& attribute_name,
    const py::dict& actor_names,
    const py::dict& vertex_matrix,
    const py::dict& edge_matrix
)
{
    auto mnet = rmnet.get_mlnet();

    if (actor_names.size() != 0)
    {
        if (vertex_matrix.size() > 0)
        {
            py::print("[Warning] unused parameter: \"vertices\"");
        }

        if (edge_matrix.size() > 0)
        {
            py::print("[Warning] unused parameter: \"edges\"");
        }

        auto actors = resolve_actors(mnet,actor_names["actor"]);
        auto attributes = mnet->actors()->attr();
        auto att = attributes->get(attribute_name);

        if (!att)
        {
            throw std::runtime_error("cannot find attribute: " + attribute_name + " for actors");
        }

        if (att->type==uu::core::AttributeType::DOUBLE)
        {
            py::list value;

            for (auto actor: actors)
            {
                auto val = attributes->get_double(actor,att->name);
                if (val.null)
                    value.append(py::none());
                else value.append(val.value);
                
            }
            py::dict res;
            res[pybind11::cast(att->name)] = value;
            return res;
        }

        else if (att->type==uu::core::AttributeType::STRING)
        {
            py::list value;

            for (auto actor: actors)
            {
                auto val = attributes->get_string(actor,att->name);
                if (val.null)
                    value.append(py::none());
                else value.append(val.value);
            }
            py::dict res;
            res[pybind11::cast(att->name)] = value;
            return res;
        }

        else
        {
            throw std::runtime_error("attribute type not supported: " + uu::core::to_string(att->type));
        }
    }

    // local attributes: vertices
    else if (vertex_matrix.size() > 0)
    {
        if (edge_matrix.size() > 0)
        {
            py::print("[Warning] unused parameter: \"edges\"");
        }

        auto vertices = resolve_vertices(mnet,vertex_matrix);

        // Get attribute type
        const uu::core::Attribute* att;
        std::set<const G*> layers;
        std::set<uu::core::AttributeType> types;
        for (auto v: vertices)
        {
            auto layer = v.second;
            layers.insert(layer);
        }
        for (auto l: layers)
        {
            auto attributes = l->vertices()->attr();
            att = attributes->get(attribute_name);
            if (att)
            {
                   types.insert(att->type);
            }
        }
        if (types.size() == 0)
        {
            throw std::runtime_error("vertex attribute " + attribute_name + " not found for the input layers");
        }
        if (types.size() > 1)
        {
            throw std::runtime_error("different attribute types on different layers");
        }
        
        auto attribute_type = *types.begin();
        
        if (attribute_type==uu::core::AttributeType::NUMERIC || attribute_type==uu::core::AttributeType::DOUBLE)
        {
            py::list value;
            
            for (size_t i = 0; i<vertices.size(); i++)
            {
                auto vertex = vertices.at(i);
                auto actor = vertex.first;
                auto layer = vertex.second;
                auto attributes = layer->vertices()->attr();
                att = attributes->get(attribute_name);
                if (!att)
                {
                    value.append(std::numeric_limits<double>::quiet_NaN());
                }
                else
                {
                    auto att_val = attributes->get_double(actor, attribute_name);
                    if (att_val.null) value.append(std::numeric_limits<double>::quiet_NaN());
                    else value.append(att_val.value);
                }
            }
            py::dict res;
            res[pybind11::cast(attribute_name)] = value;
            return res;
        }
        
        else if (attribute_type==uu::core::AttributeType::STRING)
        {
            py::list value;
            
            for (size_t i = 0; i<vertices.size(); i++)
            {
                auto vertex = vertices.at(i);
                auto actor = vertex.first;
                auto layer = vertex.second;
                auto attributes = layer->vertices()->attr();
                att = attributes->get(attribute_name);
                if (!att)
                {
                    value.append("");
                }
                else
                {
                    auto att_val = attributes->get_string(actor, attribute_name);
                    if (att_val.null) value.append(py::none());
                    else value.append(att_val.value);
                }
            }
            py::dict res;
            res[pybind11::cast(attribute_name)] = value;
            return res;
        }

        else
        {
            throw std::runtime_error("attribute type not supported: " + uu::core::to_string(attribute_type));
        }
    }

    else if (edge_matrix.size() > 0)
    {
        auto edges = resolve_edges(mnet,edge_matrix);
        
        // Get attribute type
        const uu::core::Attribute* att;
        std::set<std::pair<const G*,const G*>> layers;
        std::set<uu::core::AttributeType> types;
        for (auto edge: edges)
        {
            auto layer1 = std::get<1>(edge);
            auto layer2 = std::get<3>(edge);
            layers.insert(std::pair<const G*,const G*>(layer1, layer2));
        }
        for (auto p: layers)
        {
            auto layer1 = p.first;
            auto layer2 = p.second;
            if (layer1 == layer2)
            {
                auto attributes = layer1->edges()->attr();
                att = attributes->get(attribute_name);
                if (att)
                {
                    types.insert(att->type);
                }
            }
            else
            {
                auto attributes = mnet->interlayer_edges()->attr();
                att = attributes->get(attribute_name);
                if (att)
                {
                    types.insert(att->type);
                }
            }
        }
        if (types.size() == 0)
        {
            throw std::runtime_error("edge attribute " + attribute_name + " not found for the input layers");
        }
        if (types.size() > 1)
        {
            throw std::runtime_error("different attribute types on different combinations of layers");
        }
        
        auto attribute_type = *types.begin();
        
        if (attribute_type == uu::core::AttributeType::DOUBLE)
        {
            py::list value;

            for (auto edge: edges)
            {
                auto actor1 = std::get<0>(edge);
                auto layer1 = std::get<1>(edge);
                auto actor2 = std::get<2>(edge);
                auto layer2 = std::get<3>(edge);
                if (layer1 == layer2)
                {
                    auto attributes = layer1->edges()->attr();
                    if (!attributes->get(attribute_name))
                    {
                        value.append(std::numeric_limits<double>::quiet_NaN());
                    }
                    else
                    {
                        auto e = layer1->edges()->get(actor1, actor2);
                        auto val = attributes->get_double(e, attribute_name);
                        if (val.null)
                            value.append(py::none());
                        else value.append(val.value);
                    }
                }
                else
                {
                    auto attributes = mnet->interlayer_edges()->attr();
                    auto e = mnet->interlayer_edges()->get(actor1, layer1, actor2, layer2);
                    auto att_val = attributes->get_double(e, attribute_name);
                    if (att_val.null) value.append(std::numeric_limits<double>::quiet_NaN());
                    else value.append(att_val.value);
                }
            }
            py::dict res;
            res[pybind11::cast(attribute_name)] = value;
            return res;
        }

        else if (attribute_type == uu::core::AttributeType::STRING)
        {
            py::list value;

            for (auto edge: edges)
            {
                auto actor1 = std::get<0>(edge);
                auto layer1 = std::get<1>(edge);
                auto actor2 = std::get<2>(edge);
                auto layer2 = std::get<3>(edge);
                if (layer1 == layer2)
                {
                    auto attributes = layer1->edges()->attr();
                    if (!attributes->get(attribute_name))
                    {
                        value.append("");
                    }
                    else
                    {
                        auto e = layer1->edges()->get(actor1, actor2);
                        auto val = attributes->get_string(e, attribute_name);
                        if (val.null)
                            value.append(py::none());
                        else value.append(val.value);
                    }
                }
                else
                {
                    auto attributes = mnet->interlayer_edges()->attr();
                    auto e = mnet->interlayer_edges()->get(actor1, layer1, actor2, layer2);
                    auto att_val = attributes->get_string(e, attribute_name);
                    if (att_val.null) value.append(py::none());
                    else value.append(att_val.value);
                }
            }
            py::dict res;
            res[pybind11::cast(attribute_name)] = value;
            return res;
        }

        else
        {
            throw std::runtime_error("attribute type not supported: " + uu::core::to_string(attribute_type));
        }
    }

    else
    {
        throw std::runtime_error("Required at least one parameter: \"actors\", \"vertices\" or \"edges\"");
    }

    return py::dict();
}

void
setValues(
    PyMLNetwork& rmnet,
    const std::string& attribute_name,
    const py::dict& actor_names,
    const py::dict& vertex_matrix,
    const py::dict& edge_matrix,
    const py::list& values
)
{
    auto mnet = rmnet.get_mlnet();
    
    
    if (actor_names.size() != 0)
    {
        auto actors = resolve_actors(mnet,actor_names["actor"]);
        
        if (actors.size() != values.size() && values.size()!=1)
        {
            throw std::runtime_error("wrong number of values");
        }

        if (vertex_matrix.size() > 0)
        {
            py::print("[Warning] unused parameter: \"vertices\"");
        }

        if (edge_matrix.size() > 0)
        {
            py::print("[Warning] unused parameter: \"edges\"");
        }

        auto attributes = mnet->actors()->attr();
        auto att = attributes->get(attribute_name);

        if (!att)
        {
            throw std::runtime_error("cannot find attribute: " + attribute_name + " for actors");
        }

        size_t i=0;

        for (auto actor: actors)
        {
            switch (att->type)
            {
            case uu::core::AttributeType::NUMERIC:
            case uu::core::AttributeType::DOUBLE:
                if (values.size()==1)
                {
                    attributes->set_double(actor,att->name,values[0].cast<double>());
                }

                else
                {
                    attributes->set_double(actor,att->name,values[i].cast<double>());
                }

                break;

            case uu::core::AttributeType::STRING:
                if (values.size()==1)
                {
                    attributes->set_string(actor,att->name,values[0].cast<std::string>());
                }

                else
                {
                    attributes->set_string(actor,att->name,values[i].cast<std::string>());
                }

                break;

            case uu::core::AttributeType::TEXT:
            case uu::core::AttributeType::TIME:
            case uu::core::AttributeType::INTEGER:
            case uu::core::AttributeType::INTEGERSET:
            case uu::core::AttributeType::DOUBLESET:
            case uu::core::AttributeType::STRINGSET:
            case uu::core::AttributeType::TIMESET:
                throw std::runtime_error("attribute type not supported: " + uu::core::to_string(att->type));

            }

            i++;
        }
    }

    // local attributes: vertices
    else if (vertex_matrix.size() > 0)
    {
        if (edge_matrix.size() > 0)
        {
            py::print("[Warning] unused parameter: \"edges\"");
        }

        auto vertices = resolve_vertices(mnet,vertex_matrix);

        if (vertices.size() != values.size() && values.size()!=1)
        {
            throw std::runtime_error("wrong number of values");
        }

        size_t i=0;

        for (auto vertex: vertices)
        {
            auto attributes = vertex.second->vertices()->attr();
            auto att = attributes->get(attribute_name);

            if (!att)
            {
                throw std::runtime_error("cannot find attribute: " + attribute_name + " for vertices on layer " + vertex.second->name);
            }

            switch (att->type)
            {
            case uu::core::AttributeType::NUMERIC:
            case uu::core::AttributeType::DOUBLE:
                if (values.size()==1)
                {
                    attributes->set_double(vertex.first,att->name,values[0].cast<double>());
                }

                else
                {
                    attributes->set_double(vertex.first,att->name,values[i].cast<double>());
                }

                    i++;
                    
                break;

            case uu::core::AttributeType::STRING:
                if (values.size()==1)
                {
                    attributes->set_string(vertex.first,att->name,values[0].cast<std::string>());
                }

                else
                {
                    attributes->set_string(vertex.first,att->name,values[i].cast<std::string>());
                }

                    i++;
                    
                break;

            case uu::core::AttributeType::TEXT:
            case uu::core::AttributeType::TIME:
            case uu::core::AttributeType::INTEGER:
            case uu::core::AttributeType::INTEGERSET:
            case uu::core::AttributeType::DOUBLESET:
            case uu::core::AttributeType::STRINGSET:
            case uu::core::AttributeType::TIMESET:
                throw std::runtime_error("attribute type not supported: " + uu::core::to_string(att->type));

            }

        }
    }

    else if (edge_matrix.size() > 0)
    {
        auto edges = resolve_edges(mnet,edge_matrix);

        if (edges.size() != values.size() && values.size()!=1)
        {
            throw std::runtime_error("wrong number of values");
        }

        size_t i=0;

        for (auto edge: edges)
        {
         
         auto actor1 = std::get<0>(edge);
         auto layer1 = std::get<1>(edge);
         auto actor2 = std::get<2>(edge);
         auto layer2 = std::get<3>(edge);
         
            if (layer1 == layer2)
            {
                
            
            auto attributes = layer1->edges()->attr();
            auto att = attributes->get(attribute_name);
                auto e = layer1->edges()->get(actor1, actor2);

            if (!att)
            {
                throw std::runtime_error("cannot find attribute: " + attribute_name + " for edges on layer " + layer1->name);
            }

            switch (att->type)
            {
            case uu::core::AttributeType::NUMERIC:
            case uu::core::AttributeType::DOUBLE:
                if (values.size()==1)
                {
                    attributes->set_double(e,att->name,values[0].cast<double>());
                }

                else
                {
                    attributes->set_double(e,att->name,values[i].cast<double>());
                }

                    i++;
                    
                break;

            case uu::core::AttributeType::STRING:
                if (values.size()==1)
                {
                    attributes->set_string(e,att->name,values[0].cast<std::string>());
                }

                else
                {
                    attributes->set_string(e,att->name,values[i].cast<std::string>());
                }

                    i++;
                    
                break;

            case uu::core::AttributeType::TEXT:
            case uu::core::AttributeType::TIME:
            case uu::core::AttributeType::INTEGER:
            case uu::core::AttributeType::INTEGERSET:
            case uu::core::AttributeType::DOUBLESET:
            case uu::core::AttributeType::STRINGSET:
            case uu::core::AttributeType::TIMESET:
                throw std::runtime_error("attribute type not supported: " + uu::core::to_string(att->type));

            }
            }

            else
            {
                
                auto attributes = mnet->interlayer_edges()->attr();
                auto att = attributes->get(attribute_name);
                auto e = mnet->interlayer_edges()->get(actor1, layer1, actor2, layer2);
                
                if (!att)
                {
                    throw std::runtime_error("cannot find attribute: " + attribute_name + " for edges on layers " + layer1->name +
                         ", " + layer2->name);
                }
                
                switch (att->type)
                {
                    case uu::core::AttributeType::NUMERIC:
                    case uu::core::AttributeType::DOUBLE:
                    if (values.size()==1)
                    {
                        attributes->set_double(e,att->name,values[0].cast<double>());
                    }
                    
                    else
                    {
                        attributes->set_double(e,att->name,values[i].cast<double>());
                    }
                    
                        i++;
                        
                    break;
                    
                    case uu::core::AttributeType::STRING:
                    if (values.size()==1)
                    {
                        attributes->set_string(e,att->name,values[0].cast<std::string>());
                    }
                    
                    else
                    {
                        attributes->set_string(e,att->name,values[i].cast<std::string>());
                    }
                    
                        i++;
                        
                    break;
                    
                    case uu::core::AttributeType::TEXT:
                    case uu::core::AttributeType::TIME:
                    case uu::core::AttributeType::INTEGER:
                    case uu::core::AttributeType::INTEGERSET:
                    case uu::core::AttributeType::DOUBLESET:
                    case uu::core::AttributeType::STRINGSET:
                    case uu::core::AttributeType::TIMESET:
                    throw std::runtime_error("attribute type not supported: " + uu::core::to_string(att->type));
                    
                }
            }
        }
    }

    else
    {
        throw std::runtime_error("Required at least one parameter: \"actors\", \"vertices\" or \"edges\"");
    }
}

// TRANSFORMATION

void
flatten(
    PyMLNetwork& rmnet,
    const std::string& new_layer_name,
    const py::list& layer_names,
    const std::string& method,
    bool force_directed,
    bool all_actors
)
{

    // @todo
    if (all_actors)
    {
        throw std::runtime_error("option to include all actors not currently implemented");
    }

    auto mnet = rmnet.get_mlnet();

    auto layers = resolve_layers_unordered(mnet,layer_names);

    bool directed = force_directed;

    if (!force_directed)
    {
        for (auto layer: layers)
        {
            if (layer->is_directed())
            {
                directed = true;
                break;
            }
        }
    }

    auto edge_directionality = directed?uu::net::EdgeDir::DIRECTED:uu::net::EdgeDir::UNDIRECTED;


    auto target = mnet->layers()->add(new_layer_name, edge_directionality, uu::net::LoopMode::ALLOWED);
    target->edges()->attr()->add("weight", uu::core::AttributeType::DOUBLE);
    
    if (method=="weighted")
    {
        uu::net::weighted_graph_union(layers.begin(),layers.end(),target,"weight");
    }

    else if (method=="or")
    {
        // todo replace with new union
        for (auto g=layers.begin(); g!=layers.end(); ++g)
        {
            uu::net::graph_add(*g, target);
        }
    }

    else
    {
        throw std::runtime_error("Unexpected value: method");
    }
}


void project(
    PyMLNetwork& rmnet,
    const std::string& new_layer,
    const std::string& layer_name1,
    const std::string& layer_name2,
    const std::string& method)
{
auto mnet = rmnet.get_mlnet();
auto layer1 = mnet->layers()->get(layer_name1);
auto layer2 = mnet->layers()->get(layer_name2);
if (!layer1 || !layer2)
    throw std::runtime_error("Layer not found");
    if (method=="clique")
    {
        auto target_ptr = mnet->layers()->add(new_layer, uu::net::EdgeDir::UNDIRECTED, uu::net::LoopMode::ALLOWED);
        uu::net::project_unweighted(mnet, layer1, layer2, target_ptr);
    }
    else throw std::runtime_error("Unexpected value: method");
}

// MEASURES

py::list
degree_ml(
    const PyMLNetwork& rmnet,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type
)
{
    auto mnet = rmnet.get_mlnet();

    auto actors = resolve_actors(mnet,actor_names);
    auto layers = resolve_layers_unordered(mnet,layer_names);
    py::list res;

    for (auto actor: actors)
    {
        long deg = 0;
        auto mode = resolve_mode(type);
        deg = degree(layers.begin(), layers.end(), actor, mode);

        if (deg==0)
        {
            // check if the actor is missing from all layer_names
            bool is_missing = true;

            for (auto layer: layers)
            {
                if (layer->vertices()->contains(actor))
                {
                    is_missing = false;
                }
            }

            if (is_missing)
            {
                res.append(NAN);
            }

            else
            {
                res.append(0);
            }
        }

        else
        {
            res.append(deg);
        }
    }

    return res;
}


py::list
degree_deviation_ml(
    const PyMLNetwork& rmnet,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type)
{
    auto mnet = rmnet.get_mlnet();

    auto actors = resolve_actors(mnet,actor_names);
    auto layers = resolve_layers_unordered(mnet,layer_names);
    py::list res;

    for (auto actor: actors)
    {
        double deg = 0;
        auto mode = resolve_mode(type);
        deg = degree_deviation(layers.begin(), layers.end(), actor, mode);

        if (deg==0)
        {
            // check if the actor is missing from all layer_names
            bool is_missing = true;

            for (auto layer: layers)
            {
                if (layer->vertices()->contains(actor))
                {
                    is_missing = false;
                }
            }

            if (is_missing)
            {
                res.append(NAN);
            }

            else
            {
                res.append(0);
            }
        }

        else
        {
            res.append(deg);
        }
    }

    return res;
}


py::list
neighborhood_ml(
    const PyMLNetwork& rmnet,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type
)
{
    auto mnet = rmnet.get_mlnet();

    auto actors = resolve_actors(mnet,actor_names);
    auto layers = resolve_layers_unordered(mnet,layer_names);
    py::list res;

    for (auto actor: actors)
    {
        long neigh = 0;
        auto mode = resolve_mode(type);
        neigh = neighbors(layers.begin(), layers.end(), actor, mode).size();

        if (neigh==0)
        {
            // check if the actor is missing from all layer_names
            bool is_missing = true;

            for (auto layer: layers)
            {
                if (layer->vertices()->contains(actor))
                {
                    is_missing = false;
                }
            }

            if (is_missing)
            {
                res.append(NAN);
            }

            else
            {
                res.append(0);
            }
        }

        else
        {
            res.append(neigh);
        }
    }

    return res;
}



py::list
xneighborhood_ml(
    const PyMLNetwork& rmnet,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type)
{
    auto mnet = rmnet.get_mlnet();

    auto actors = resolve_actors(mnet,actor_names);
    auto layers = resolve_layers_unordered(mnet,layer_names);
    py::list res;

    for (auto actor: actors)
    {
        long neigh = 0;
        auto mode = resolve_mode(type);
        neigh = xneighbors(mnet, layers.begin(), layers.end(), actor, mode).size();

        if (neigh==0)
        {
            // check if the actor is missing from all layer_names
            bool is_missing = true;

            for (auto layer: layers)
            {
                if (layer->vertices()->contains(actor))
                {
                    is_missing = false;
                }
            }

            if (is_missing)
            {
                res.append(NAN);
            }

            else
            {
                res.append(0);
            }
        }

        else
        {
            res.append(neigh);
        }
    }

    return res;
}

py::list
connective_redundancy_ml(
    const PyMLNetwork& rmnet,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type)
{
    auto mnet = rmnet.get_mlnet();

    auto actors = resolve_actors(mnet,actor_names);
    auto layers = resolve_layers_unordered(mnet,layer_names);
    py::list res;
    double cr = 0;

    for (auto actor: actors)
    {
        auto mode = resolve_mode(type);

        cr = uu::net::connective_redundancy(mnet, layers.begin(), layers.end(), actor, mode);

        if (cr==0)
        {
            // check if the actor is missing from all layer_names
            bool is_missing = true;

            for (auto layer: layers)
            {
                if (layer->vertices()->contains(actor))
                {
                    is_missing = false;
                }
            }

            if (is_missing)
            {
                res.append(NAN);
            }

            else
            {
                res.append(0);
            }
        }

        else
        {
            res.append(cr);
        }
    }

    return res;
}

py::list
relevance_ml(
    const PyMLNetwork& rmnet,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type)
{
    auto mnet = rmnet.get_mlnet();

    auto actors = resolve_actors(mnet,actor_names);
    auto layers = resolve_layers_unordered(mnet,layer_names);
    py::list res;

    for (auto actor: actors)
    {
        double rel = 0;
        auto mode = resolve_mode(type);
        rel = uu::net::relevance(mnet, layers.begin(), layers.end(), actor, mode);

        if (rel==0)
        {
            // check if the actor is missing from all layer_names
            bool is_missing = true;

            for (auto layer: layers)
            {
                if (layer->vertices()->contains(actor))
                {
                    is_missing = false;
                }
            }

            if (is_missing)
            {
                res.append(NAN);
            }

            else
            {
                res.append(0);
            }
        }

        else
        {
            res.append(rel);
        }
    }

    return res;
}


py::list
xrelevance_ml(
    const PyMLNetwork& rmnet,
    const py::list& actor_names,
    const py::list& layer_names,
    const std::string& type)
{
    auto mnet = rmnet.get_mlnet();

    auto actors = resolve_actors(mnet,actor_names);
    auto layers = resolve_layers_unordered(mnet,layer_names);

    py::list res;

    for (auto actor: actors)
    {
        double rel = 0;
        auto mode = resolve_mode(type);
        rel = uu::net::xrelevance(mnet, layers.begin(), layers.end(), actor, mode);

        if (rel==0)
        {
            // check if the actor is missing from all layer_names
            bool is_missing = true;

            for (auto layer: layers)
            {
                if (layer->vertices()->contains(actor))
                {
                    is_missing = false;
                }
            }

            if (is_missing)
            {
                res.append(NAN);
            }

            else
            {
                res.append(0);
            }
        }

        else
        {
            res.append(rel);
        }
    }

    return res;
}


py::list
comparison_ml(
    const PyMLNetwork& rmnet,
    const py::list& layer_names,
    const std::string& method,
    const std::string& type,
    int K
)
{

    auto mnet = rmnet.get_mlnet();
    std::vector<uu::net::Network*> layers = resolve_layers(mnet,layer_names);
    std::vector<py::list> values;

    for (size_t i=0; i<layers.size(); i++)
    {
        py::list v;
        values.push_back(v);
    }

    //py::dict res = py::dict::create();

    if (method=="jaccard.actors")
    {
        uu::core::PropertyMatrix<const uu::net::Vertex*, const uu::net::Network*,bool> P = uu::net::actor_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::jaccard<const uu::net::Vertex*, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="coverage.actors")
    {
        uu::core::PropertyMatrix<const uu::net::Vertex*, const uu::net::Network*,bool> P = uu::net::actor_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::coverage<const uu::net::Vertex*, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="kulczynski2.actors")
    {
        uu::core::PropertyMatrix<const uu::net::Vertex*, const uu::net::Network*,bool> P = uu::net::actor_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::kulczynski2<const uu::net::Vertex*, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="sm.actors")
    {
        uu::core::PropertyMatrix<const uu::net::Vertex*, const uu::net::Network*,bool> P = uu::net::actor_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::simple_matching<const uu::net::Vertex*, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="rr.actors")
    {
        uu::core::PropertyMatrix<const uu::net::Vertex*, const uu::net::Network*,bool> P = uu::net::actor_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::russell_rao<const uu::net::Vertex*, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="hamann.actors")
    {
        uu::core::PropertyMatrix<const uu::net::Vertex*, const uu::net::Network*,bool> P = uu::net::actor_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::hamann<const uu::net::Vertex*, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="jaccard.edges")
    {
        auto P = uu::net::edge_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::jaccard<std::pair<const typename M::vertex_type*,const typename M::vertex_type*>, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="coverage.edges")
    {
        auto P = uu::net::edge_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::coverage<std::pair<const typename M::vertex_type*,const typename M::vertex_type*>, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="kulczynski2.edges")
    {
        auto P = uu::net::edge_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::kulczynski2<std::pair<const typename M::vertex_type*,const typename M::vertex_type*>, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="sm.edges")
    {
        auto P = uu::net::edge_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::simple_matching<std::pair<const typename M::vertex_type*,const typename M::vertex_type*>, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="rr.edges")
    {
        auto P = uu::net::edge_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::russell_rao<std::pair<const typename M::vertex_type*,const typename M::vertex_type*>, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="hamann.edges")
    {
        auto P = uu::net::edge_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::hamann<std::pair<const typename M::vertex_type*,const typename M::vertex_type*>, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="jaccard.triangles")
    {
        uu::core::PropertyMatrix<uu::net::Triad, const uu::net::Network*,bool> P = uu::net::triangle_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::jaccard<uu::net::Triad, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="coverage.triangles")
    {
        uu::core::PropertyMatrix<uu::net::Triad, const uu::net::Network*,bool> P = uu::net::triangle_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::coverage<uu::net::Triad, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="kulczynski2.triangles")
    {
        uu::core::PropertyMatrix<uu::net::Triad, const uu::net::Network*,bool> P = uu::net::triangle_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::kulczynski2<uu::net::Triad, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="sm.triangles")
    {
        uu::core::PropertyMatrix<uu::net::Triad, const uu::net::Network*,bool> P = uu::net::triangle_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::simple_matching<uu::net::Triad, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="rr.triangles")
    {
        uu::core::PropertyMatrix<uu::net::Triad, const uu::net::Network*,bool> P = uu::net::triangle_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::russell_rao<uu::net::Triad, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="hamann.triangles")
    {
        uu::core::PropertyMatrix<uu::net::Triad, const uu::net::Network*,bool> P = uu::net::triangle_existence_property_matrix(mnet);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::hamann<uu::net::Triad, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="dissimilarity.degree")
    {
        auto mode = resolve_mode(type);
        uu::core::PropertyMatrix<const uu::net::Vertex*, const uu::net::Network*,double> P = uu::net::actor_degree_property_matrix(mnet,mode);

        if (K<=0)
        {
            K=std::ceil(std::log2(P.num_structures) + 1);
        }

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::dissimilarity_index<const uu::net::Vertex*, const uu::net::Network*>(P,layers[i],layers[j],K));
            }
        }
    }

    else if (method=="KL.degree")
    {
        auto mode = resolve_mode(type);
        uu::core::PropertyMatrix<const uu::net::Vertex*, const uu::net::Network*,double> P = uu::net::actor_degree_property_matrix(mnet,mode);

        if (K<=0)
        {
            K=std::ceil(std::log2(P.num_structures) + 1);
        }

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::KL_divergence<const uu::net::Vertex*, const uu::net::Network*>(P,layers[i],layers[j],K));
            }
        }
    }

    else if (method=="jeffrey.degree")
    {
        auto mode = resolve_mode(type);
        uu::core::PropertyMatrix<const uu::net::Vertex*, const uu::net::Network*,double> P = uu::net::actor_degree_property_matrix(mnet,mode);

        if (K<=0)
        {
            K=std::ceil(std::log2(P.num_structures) + 1);
        }

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::jeffrey_divergence<const uu::net::Vertex*, const uu::net::Network*>(P,layers[i],layers[j],K));
            }
        }
    }

    else if (method=="pearson.degree")
    {
        auto mode = resolve_mode(type);
        uu::core::PropertyMatrix<const uu::net::Vertex*, const uu::net::Network*,double> P = uu::net::actor_degree_property_matrix(mnet,mode);

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::pearson<const uu::net::Vertex*, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else if (method=="rho.degree")
    {
        auto mode = resolve_mode(type);
        uu::core::PropertyMatrix<const uu::net::Vertex*, const uu::net::Network*,double> P = uu::net::actor_degree_property_matrix(mnet,mode);
        P.rankify();

        for (size_t j=0; j<layers.size(); j++)
        {
            for (size_t i=0; i<layers.size(); i++)
            {
                values[j].append(uu::core::pearson<const uu::net::Vertex*, const uu::net::Network*>(P,layers[i],layers[j]));
            }
        }
    }

    else
    {
        throw std::runtime_error("Unexpected value: method parameter");
    }

    /*
    if (layer_names.size()==0)
    {
        py::list names;

        for (auto l: layers)
        {
            names.append(l->name);
        }

        for (size_t i=0; i<layers.size(); i++)
        {
            res.append(values[i],std::string(names[i]));
        }

        res.attr("class") = "data.frame";
        res.attr("row.names") = names;
    }

    else
    {
        for (size_t i=0; i<layers.size(); i++)
        {
            res.append(values[i],std::string(layer_names[i]));
        }

        res.attr("class") = "data.frame";
        res.attr("row.names") = layer_names;
    }
     */

    
    py::list res;
    
    for (auto l: values)
    {
        res.append(l);
    }
    
    return res;
}

double
summary_ml(
    const PyMLNetwork& rmnet,
    const std::string& layer_name,
    const std::string& method,
    const std::string& type
)
{

    auto mnet = rmnet.get_mlnet();
    auto layer = mnet->layers()->get(layer_name);

    if (!layer)
    {
        throw std::runtime_error("no layer named " + layer_name);
    }

    auto mode = resolve_mode(type);
    uu::core::PropertyMatrix<const uu::net::Vertex*, const uu::net::Network*,double> P = uu::net::actor_degree_property_matrix(mnet,mode);

    if (method=="min.degree")
    {
        return uu::core::min<const uu::net::Vertex*, const uu::net::Network*>(P,layer);
    }

    else if (method=="max.degree")
    {
        return uu::core::max<const uu::net::Vertex*, const uu::net::Network*>(P,layer);
    }

    else if (method=="sum.degree")
    {
        return uu::core::sum<const uu::net::Vertex*, const uu::net::Network*>(P,layer);
    }

    else if (method=="mean.degree")
    {
        return uu::core::mean<const uu::net::Vertex*, const uu::net::Network*>(P,layer);
    }

    else if (method=="sd.degree")
    {
        return uu::core::sd<const uu::net::Vertex*, const uu::net::Network*>(P,layer);
    }

    else if (method=="skewness.degree")
    {
        return uu::core::skew<const uu::net::Vertex*, const uu::net::Network*>(P,layer);
    }

    else if (method=="kurtosis.degree")
    {
        return uu::core::kurt<const uu::net::Vertex*, const uu::net::Network*>(P,layer);
    }

    else if (method=="entropy.degree")
    {
        return uu::core::entropy<const uu::net::Vertex*, const uu::net::Network*>(P,layer);
    }

    else if (method=="CV.degree")
    {
        return uu::core::CV<const uu::net::Vertex*, const uu::net::Network*>(P,layer);
    }

    else if (method=="jarque.bera.degree")
    {
        return uu::core::jarque_bera<const uu::net::Vertex*, const uu::net::Network*>(P,layer);
    }

    else
    {
        throw std::runtime_error("Unexpected value: method parameter");
    }

    return 0;
}



py::dict
distance_ml(
    const PyMLNetwork& rmnet,
    const std::string& from_actor,
    const py::list& to_actors,
    const std::string& method
    )
{
    auto mnet = rmnet.get_mlnet();
    std::vector<const uu::net::Vertex*> actors_to = resolve_actors(mnet,to_actors);
    auto actor_from = mnet->actors()->get(from_actor);

    py::dict res;
    
    if (!actor_from)
    {
        throw std::runtime_error("no actor named " + from_actor);
    }

    if (method=="multiplex")
    {
        auto dists = uu::net::pareto_distance(mnet, actor_from);

        py::list from, to;
        std::vector<py::list> lengths;

        for (size_t i=0; i<mnet->layers()->size(); i++)
        {
            py::list v;
            lengths.push_back(v);
        }

        for (auto actor: actors_to)
        {
            for (auto d: dists[actor])
            {
                from.append(from_actor);
                to.append(actor->name);

                for (size_t i=0; i<mnet->layers()->size(); i++)
                {
                    lengths[i].append(d.length(mnet->layers()->at(i)));
                }
            }
        }

        res["from"] = from;
        res["to"] = to;

        for (size_t i=0; i<mnet->layers()->size(); i++)
        {
            res[mnet->layers()->at(i)->name.c_str()] = lengths[i];
        }
    }

    else
    {
        throw std::runtime_error("Unexpected value: method");
    }

    return res;
}


/*
NumericMatrix sir_ml(
    const PyMLNetwork& rmnet, double beta, int tau, long num_iterations) {
auto mnet = rmnet.get_mlnet();
matrix<long> stats = sir(mnet, beta, tau, num_iterations);

NumericMatrix res(3,num_iterations+1);

py::list colnames(0);
py::list rownames(3);
rownames(0) = "S";
rownames(1) = "I";
rownames(2) = "R";
res.attr("dimnames") = List::create(rownames, colnames);

for (size_t i=0; i<3; i++) {
for (long j=0; j<num_iterations+1; j++) {
res(i,j) = stats[i][j];
}
}
return res;
} ///////
*/

// COMMUNITY DETECTION

py::dict
cliquepercolation_ml(
    const PyMLNetwork& rmnet,
    int k,
    int m
)
{
    auto mnet = rmnet.get_mlnet();

    auto com_struct = mlcpm(mnet, k, m);
    return to_dataframe(com_struct.get());
}


py::dict
glouvain_ml(
    const PyMLNetwork& rmnet,
    double gamma,
    double omega
)
{
    auto mnet = rmnet.get_mlnet();

    auto com_struct = uu::net::glouvain2<M>(mnet, omega, gamma);

    return to_dataframe(com_struct.get());
}

py::dict
infomap_ml(const PyMLNetwork& rmnet,
           bool overlapping,
           bool directed,
           bool include_self_links
          )
{
    auto mnet = rmnet.get_mlnet();

    try
    {
        auto com_struct = uu::net::infomap(mnet, overlapping, directed, include_self_links);
        return to_dataframe(com_struct.get());
    }

    catch (std::exception& e)
    {
        py::print("[Warning] could not run external library: " + std::string(e.what()));
        py::print("Returning empty community set.");
    }

    auto com_struct = std::make_unique<uu::net::CommunityStructure<uu::net::MultilayerNetwork>>();
    return to_dataframe(com_struct.get());
}

py::dict
abacus_ml(
    const PyMLNetwork& rmnet,
    int min_actors,
    int min_layers
)
{
    auto mnet = rmnet.get_mlnet();

    try
    {
        auto com_struct = uu::net::abacus(mnet, min_actors, min_layers);
        return to_dataframe(com_struct.get());
    }

    catch (std::exception& e)
    {
        py::print("[Warning] could not run external library: " + std::string(e.what()));
        py::print("Returning empty community set.");
    }

    auto com_struct = std::make_unique<uu::net::CommunityStructure<uu::net::MultilayerNetwork>>();
    return to_dataframe(com_struct.get());

}

py::dict
flat_ec(
    const PyMLNetwork& rmnet
)
{
    auto mnet = rmnet.get_mlnet();

    auto com_struct = uu::net::flat_ec(mnet);
    return to_dataframe(com_struct.get());
}

py::dict
flat_nw(
    const PyMLNetwork& rmnet
)
{
    auto mnet = rmnet.get_mlnet();

    auto com_struct = uu::net::flat_nw(mnet);
    return to_dataframe(com_struct.get());
}

py::dict
mdlp(
     const PyMLNetwork& rmnet
)
{
    auto mnet = rmnet.get_mlnet();

    auto com_struct = uu::net::mlp(mnet);
    return to_dataframe(com_struct.get());
}


double
modularity_ml(
              const PyMLNetwork& rmnet,
              const py::dict& com,
              double gamma,
              double omega
              )
{
    auto mnet = rmnet.get_mlnet();
    auto communities = to_communities(com, mnet);
    return uu::net::modularity(mnet, communities.get(), omega);
}

double
nmi(
    const PyMLNetwork& rmnet,
    const py::dict& com1,
    const py::dict& com2
)
{
    size_t num_vertices = numNodes(rmnet, py::list());
    auto mnet = rmnet.get_mlnet();
    auto c1 = to_communities(com1, mnet);
    auto c2 = to_communities(com2, mnet);
    return uu::net::nmi(c1.get(), c2.get(), num_vertices);
}

double
omega(
    const PyMLNetwork& rmnet,
    const py::dict& com1,
    const py::dict& com2
)
{
    size_t num_vertices = numNodes(rmnet, py::list());
    auto mnet = rmnet.get_mlnet();
    auto c1 = to_communities(com1, mnet);
    auto c2 = to_communities(com2, mnet);
    return uu::net::omega_index(c1.get(), c2.get(), num_vertices);
}

/*
List
to_list(
    const py::dict& cs,
    const PyMLNetwork& rmnet
)
{
    auto mnet = rmnet.get_mlnet();

    // stores at which index vertices start in a layer
    std::unordered_map<const G*, size_t> offset;
    size_t num_vertices = 0;

    for (auto layer: *mnet->layers())
    {
        offset[layer] = num_vertices;
        num_vertices += layer->vertices()->size();
    }


    std::map<int, std::map<int, std::vector<int> > > list;
    py::list cs_actor = cs["actor"];
    py::list cs_layer = cs["layer"];
    py::list cs_cid = cs["cid"];

    for (size_t i=0; i<cs.nrow(); i++)
    {
        int comm_id = cs_cid[i];
        auto layer = mnet->layers()->get(std::string(cs_layer[i]));

        if (!layer)
        {
            throw std::runtime_error("cannot find layer " + std::string(cs_layer[i]) + " (community structure not compatible with this network?)");
        }

        int l = mnet->layers()->index_of(layer);
        auto actor = mnet->actors()->get(std::string(cs_actor[i]));

        if (!actor)
        {
            throw std::runtime_error("cannot find actor " + std::string(cs_actor[i]) + " (community structure not compatible with this network?)");
        }

        int vertex_idx = layer->vertices()->index_of(actor);

        if (vertex_idx==-1)
        {
            throw std::runtime_error("cannot find vertex " + std::string(cs_actor[i]) + "::" + std::string(cs_layer[i]) + " (community structure not compatible with this network?)");
        }

        int n = vertex_idx+offset[layer]+1;
        list[comm_id][l].append(n);
    }

    List res = List::create();

    for (auto clist: list)
    {
        for (auto llist: clist.second)
        {
            res.append(List::create(res["cid"]=clist.first, res["lid"]=llist.first, res["aid"]=llist.second));
        }
    }

    return res;
}
*/

// LAYOUT

py::dict
multiforce_ml(
    const PyMLNetwork& rmnet,
    const py::list& w_in,
    const py::list& w_inter,
    const py::list& gravity,
    int iterations
)
{
    auto mnet = rmnet.get_mlnet();
    std::unordered_map<const G*,double> weight_in, weight_inter, weight_gr;
    auto layers = mnet->layers();

    if (w_in.size()==1)
    {
        for (size_t i=0; i<layers->size(); i++)
        {
            weight_in[layers->at(i)] = w_in[0].cast<double>();
        }
    }

    else if (w_in.size()==layers->size())
    {
        for (size_t i=0; i<layers->size(); i++)
        {
            weight_in[layers->at(i)] = w_in[i].cast<double>();
        }
    }

    else
    {
        throw std::runtime_error("wrong dimension: internal weights (should contain 1 or num.layers.ml weights)");
    }

    if (w_inter.size()==1)
    {
        for (size_t i=0; i<layers->size(); i++)
        {
            weight_inter[layers->at(i)] = w_inter[0].cast<double>();
        }
    }

    else if (w_inter.size()==layers->size())
    {
        for (size_t i=0; i<layers->size(); i++)
        {
            weight_inter[layers->at(i)] = w_inter[i].cast<double>();
        }
    }

    else
    {
        throw std::runtime_error("wrong dimension: external weights (should contain 1 or num.layers.ml weights)");
    }

    if (gravity.size()==1)
    {
        for (size_t i=0; i<layers->size(); i++)
        {
            weight_gr[layers->at(i)] = gravity[0].cast<double>();
        }
    }

    else if (gravity.size()==layers->size())
    {
        for (size_t i=0; i<layers->size(); i++)
        {
            weight_gr[layers->at(i)] = gravity[i].cast<double>();
        }
    }

    else
    {
        throw std::runtime_error("wrong dimension: gravity (should contain 1 or num.layers.ml weights)");
    }

    auto coord = uu::net::multiforce(mnet, 10, 10, weight_in, weight_inter, weight_gr, iterations);

    std::unordered_map<const G*, size_t> offset;
    size_t num_rows = 0;

    for (auto layer: *mnet->layers())
    {
        num_rows += layer->vertices()->size();
    }

    py::list actor_n;
    py::list layer_n;
    py::list x_n;
    py::list y_n;
    py::list z_n;

    for (auto l: *mnet->layers())
    {
        for (auto a: *l->vertices())
        {

            auto n = std::make_pair(a, l);
            actor_n.append(a->name);
            layer_n.append(l->name);
            auto c = coord.at(n);
            x_n.append(c.x);
            y_n.append(c.y);
            z_n.append(c.z);
        }
    }

    py::dict vertices;
    vertices["actor"]=actor_n;
    vertices["layer"]=layer_n;
    vertices["x"]=x_n;
    vertices["y"]=y_n;
    vertices["z"]=z_n;

    return vertices;
}


py::dict
circular_ml(
    const PyMLNetwork& rmnet)
{
    auto mnet = rmnet.get_mlnet();

    auto coord = uu::net::circular(mnet, 10.0);

    std::unordered_map<const G*, size_t> offset;
    size_t num_rows = 0;

    for (auto layer: *mnet->layers())
    {
        num_rows += layer->vertices()->size();
    }

    py::list actor_n;
    py::list layer_n;
    py::list x_n;
    py::list y_n;
    py::list z_n;

    for (auto l: *mnet->layers())
    {
        for (auto a: *l->vertices())
        {

            auto n = std::make_pair(a, l);
            actor_n.append(a->name);
            layer_n.append(l->name);
            auto c = coord.at(n);
            x_n.append(c.x);
            y_n.append(c.y);
            z_n.append(c.z);
        }
    }


    py::dict vertices;
    vertices["actor"]=actor_n;
    vertices["layer"]=layer_n;
    vertices["x"]=x_n;
    vertices["y"]=y_n;
    vertices["z"]=z_n;
    
    return vertices;
}

py::dict
toNetworkxEdgeDict(
                 const PyMLNetwork& rmnet
                 )
{
    auto mnet = rmnet.get_mlnet();
    
    py::dict res;
    for (auto l: *mnet->layers())
    {
        py::dict layer_dict;
        
        for (auto v: *l->vertices())
        {
            layer_dict[v->name.c_str()] = py::dict();
        }
        
        auto edge_attrs = l->edges()->attr();
        for (auto e: *l->edges())
        {
            py::dict attr_values;
            for (auto attr: *edge_attrs)
            {
                switch (attr->type)
                {
                    case uu::core::AttributeType::NUMERIC:
                    case uu::core::AttributeType::DOUBLE:
                        attr_values[attr->name.c_str()] = edge_attrs->get_double(e,attr->name).value;
                        break;
                        
                    case uu::core::AttributeType::STRING:
                        attr_values[attr->name.c_str()] = edge_attrs->get_string(e,attr->name).value;
                        break;
                        
                    case uu::core::AttributeType::TIME:
                    case uu::core::AttributeType::TEXT:
                    case uu::core::AttributeType::INTEGER:
                    case uu::core::AttributeType::INTEGERSET:
                    case uu::core::AttributeType::DOUBLESET:
                    case uu::core::AttributeType::STRINGSET:
                    case uu::core::AttributeType::TIMESET:
                        break;
                }
            }
            layer_dict[e->v1->name.c_str()][e->v2->name.c_str()] = attr_values;
        }
        res[l->name.c_str()] = layer_dict;
    }
    return res;
}

py::dict
toNetworkxNodeDict(
        const PyMLNetwork& rmnet
        )
{
    auto mnet = rmnet.get_mlnet();
    
    py::dict res;
    for (auto l: *mnet->layers())
    {
        py::dict layer_dict;
        auto node_attrs = l->vertices()->attr();
        for (auto v: *l->vertices())
        {
            py::dict attr_values;
            
            // actor atributes
            auto actor_attr = mnet->actors()->attr();
            for (auto attr: *actor_attr)
            {
                switch (attr->type)
                {
                    case uu::core::AttributeType::NUMERIC:
                    case uu::core::AttributeType::DOUBLE:
                        attr_values[attr->name.c_str()] = actor_attr->get_double(v,attr->name).value;
                        break;
                        
                    case uu::core::AttributeType::STRING:
                        attr_values[attr->name.c_str()] = actor_attr->get_string(v,attr->name).value;
                        break;
                        
                    case uu::core::AttributeType::TIME:
                    case uu::core::AttributeType::TEXT:
                    case uu::core::AttributeType::INTEGER:
                    case uu::core::AttributeType::INTEGERSET:
                    case uu::core::AttributeType::DOUBLESET:
                    case uu::core::AttributeType::STRINGSET:
                    case uu::core::AttributeType::TIMESET:
                        break;
                }
            }
            
            // vertex atributes
            auto node_attrs = l->vertices()->attr();
            for (auto attr: *node_attrs)
            {
                switch (attr->type)
                {
                    case uu::core::AttributeType::NUMERIC:
                    case uu::core::AttributeType::DOUBLE:
                        attr_values[(l->name + ":" + attr->name).c_str()] = node_attrs->get_double(v,attr->name).value;
                        break;
                        
                    case uu::core::AttributeType::STRING:
                        attr_values[(l->name + ":" + attr->name).c_str()] = node_attrs->get_string(v,attr->name).value;
                        break;
                        
                    case uu::core::AttributeType::TIME:
                    case uu::core::AttributeType::TEXT:
                    case uu::core::AttributeType::INTEGER:
                    case uu::core::AttributeType::INTEGERSET:
                    case uu::core::AttributeType::DOUBLESET:
                    case uu::core::AttributeType::STRINGSET:
                    case uu::core::AttributeType::TIMESET:
                        break;
                }
            }
            
            layer_dict[v->name.c_str()] = attr_values;
        }
        res[l->name.c_str()] = layer_dict;
    }
    return res;
}

