#include "pycpp_utils.hpp"
//#include "core/exceptions/ElementNotFoundException.hpp"
#include <stdexcept>
#include <algorithm>

// @todo check dictionaries have the right fields

std::vector<uu::net::Network*>
resolve_layers(
    const uu::net::AttributedHomogeneousMultilayerNetwork* mnet,
    const py::list& names
)
{
    size_t result_size = names.size()?names.size():mnet->layers()->size();
    std::vector<uu::net::Network*> res(result_size);

    if (names.size()==0)
    {
        size_t i=0;
        for (auto layer: *mnet->layers())
        {
            res[i] = layer;
            i++;
        }
    }

    else
    {
        size_t i=0;
        for (py::handle obj: names)
        {
            std::string name = obj.attr("__str__")().cast<std::string>();
            auto layer = mnet->layers()->get(name);

            if (!layer)
            {
                throw std::runtime_error("cannot find layer " + name);
            }

            res[i] = layer;
            i++;
        }
    }

    return res;
}

std::unordered_set<uu::net::Network*>
resolve_layers_unordered(
    const uu::net::AttributedHomogeneousMultilayerNetwork* mnet,
    const py::list& names
)
{
    std::unordered_set<uu::net::Network*> res;

    if (names.size()==0)
    {
        for (auto layer: *mnet->layers())
        {
            res.insert(layer);
        }
    }

    else
    {
        for (py::handle obj: names)
        {
            std::string name = obj.attr("__str__")().cast<std::string>();
            auto layer = mnet->layers()->get(name);

            if (!layer)
            {
                throw std::runtime_error("cannot find layer " + name);
            }

            res.insert(layer);
        }
    }

    return res;
}



std::unordered_set<const uu::net::Network*>
resolve_const_layers_unordered(
    const uu::net::AttributedHomogeneousMultilayerNetwork* mnet,
    const py::list& names
)
{
    std::unordered_set<const uu::net::Network*> res;

    if (names.size()==0)
    {
        for (auto layer: *mnet->layers())
        {
            res.insert(layer);
        }
    }

    else
    {
        for (py::handle obj: names)
        {
            std::string name = obj.attr("__str__")().cast<std::string>();
            auto layer = mnet->layers()->get(name);

            if (!layer)
            {
                throw std::runtime_error("cannot find layer " + name);
            }

            res.insert(layer);
        }
    }

    return res;
}


std::vector<const uu::net::Vertex*>
resolve_actors(
    const uu::net::AttributedHomogeneousMultilayerNetwork* mnet,
    const py::list& names
)
{
    int result_size = names.size()?names.size():mnet->vertices()->size();
    std::vector<const uu::net::Vertex*> res(result_size);

    if (names.size()==0)
    {
        size_t i = 0;

        for (auto actor: *mnet->vertices())
        {
            res[i] = actor;
            i++;
        }
    }

    else
    {
        size_t i = 0;
        for (py::handle obj: names)
        {
            std::string name = obj.attr("__str__")().cast<std::string>();
            auto actor = mnet->vertices()->get(name);

            if (!actor)
            {
                throw std::runtime_error("cannot find actor " + name);
            }

            res[i] = actor;
            i++;
        }
    }

    return res;
}

std::unordered_set<const uu::net::Vertex*>
resolve_actors_unordered(
    const uu::net::AttributedHomogeneousMultilayerNetwork* mnet,
    const py::list& names
)
{
    std::unordered_set<const uu::net::Vertex*> res;

    if (names.size()==0)
    {
        for (auto actor: *mnet->vertices())
        {
            res.insert(actor);
        }
    }

    else
    {
        for (py::handle obj: names)
        {
            std::string name = obj.attr("__str__")().cast<std::string>();
            
            auto actor = mnet->vertices()->get(name);

            if (!actor)
            {
                throw std::runtime_error("cannot find actor " + name);
            }

            res.insert(actor);
        }
    }

    return res;
}

std::vector<std::pair<const uu::net::Vertex*, uu::net::Network*>>
        resolve_vertices(
            const uu::net::AttributedHomogeneousMultilayerNetwork* mnet,
            const py::dict& vertex_matrix
        )
{
    std::vector<std::string> a = vertex_matrix["actor"].cast<std::vector<std::string>>();
    std::vector<std::string> l = vertex_matrix["layer"].cast<std::vector<std::string>>();

    if (a.size() != l.size())
    {
        throw std::runtime_error("actors and layers should have the same length");
    }
    std::vector<std::pair<const uu::net::Vertex*, uu::net::Network*>> res(a.size());
    
    for (size_t i=0; i<a.size(); i++)
    {
        auto actor = mnet->vertices()->get(a.at(i));

        if (!actor)
        {
            throw std::runtime_error("cannot find actor " + a.at(i));
        }

        auto layer = mnet->layers()->get(l.at(i));

        if (!layer)
        {
            throw std::runtime_error("cannot find layer " + l.at(i));
        }

        int vertex = layer->vertices()->index_of(actor);

        if (vertex == -1)
        {
            throw std::runtime_error("cannot find actor " + actor->name + " on layer " + layer->name);
        }

        res[i] = std::make_pair(actor, layer);
    }

    return res;
}

std::vector<std::tuple<const uu::net::Vertex*, uu::net::Network*, const uu::net::Vertex*, uu::net::Network*>>
        resolve_edges(
            const uu::net::AttributedHomogeneousMultilayerNetwork* mnet,
            const py::dict& edges
        )
{
    std::vector<std::string> a_from = edges["from_actor"].cast<std::vector<std::string>>();
    std::vector<std::string> l_from = edges["from_layer"].cast<std::vector<std::string>>();
    std::vector<std::string> a_to = edges["to_actor"].cast<std::vector<std::string>>();
    std::vector<std::string> l_to = edges["to_layer"].cast<std::vector<std::string>>();

    if ((a_from.size() != l_from.size()) || (a_from.size() != a_to.size()) || (a_from.size() != l_to.size()))
    {
        throw std::runtime_error("all lists should have the same length");
    }
    
    std::vector<std::tuple<const uu::net::Vertex*, uu::net::Network*, const uu::net::Vertex*, uu::net::Network*>> res(a_from.size());
    
    for (size_t i=0; i<a_from.size(); i++)
    {
        auto actor1 = mnet->vertices()->get(std::string(a_from.at(i)));

        if (!actor1)
        {
            throw std::runtime_error("cannot find actor " + std::string(a_from.at(i)));
        }

        auto actor2 = mnet->vertices()->get(std::string(a_to.at(i)));

        if (!actor2)
        {
            throw std::runtime_error("cannot find actor " + std::string(a_to.at(i)));
        }

        auto layer1 = mnet->layers()->get(std::string(l_from.at(i)));

        if (!layer1)
        {
            throw std::runtime_error("cannot find layer " + std::string(l_from.at(i)));
        }

        auto layer2 = mnet->layers()->get(std::string(l_to.at(i)));

        if (!layer2)
        {
            throw std::runtime_error("cannot find layer " + std::string(l_to.at(i)));
        }

        if (layer1 == layer2)
        {
            auto edge = layer1->edges()->get(actor1, actor2);

            if (!edge)
            {
                throw std::runtime_error("cannot find edge from " + actor1->to_string() + " to "
                           + actor2->to_string() + " on layer " + layer1->name);
            }

            res[i] = std::tuple<const uu::net::Vertex*, uu::net::Network*, const uu::net::Vertex*, uu::net::Network*>(actor1, layer1, actor2, layer2);
            
        }

        else
        {
            auto edge = mnet->interlayer_edges()->get(actor1, layer1, actor2, layer2);
            
            if (!edge)
            {
                throw std::runtime_error("cannot find edge from " + actor1->to_string() + " on layer " +
                           layer1->name + " to " + actor2->to_string() + " on layer " + layer2->name);
            }

            res[i] = std::tuple<const uu::net::Vertex*, uu::net::Network*, const uu::net::Vertex*, uu::net::Network*>(actor1, layer1, actor2, layer2);
            
        }
    }

    return res;
}



uu::net::EdgeMode
resolve_mode(
    const std::string& mode
)
{
    if (mode=="all")
    {
        return uu::net::EdgeMode::INOUT;
    }

    else if (mode=="in")
    {
        return uu::net::EdgeMode::IN;
    }

    else if (mode=="out")
    {
        return uu::net::EdgeMode::OUT;
    }

    throw std::runtime_error("unexpected value: edge mode " + mode);

    return uu::net::EdgeMode::INOUT; // never reaches here
}

py::dict
to_dataframe(
    uu::net::CommunityStructure<uu::net::VertexLayerCommunity<const uu::net::Network>>* cs
)
{

    py::list actor, layer, community_id;

    int comm_id=0;

    for (auto com: *cs)
    {
        for (auto pair: *com)
        {
            actor.append(pair.first->name);
            layer.append(pair.second->name);
            community_id.append(comm_id);
        }

        comm_id++;
    }

    py::dict res;
    res["actor"] = actor;
    res["layer"] = layer;
    res["cid"] = community_id;
    
    
    return res;
}

std::unique_ptr<uu::net::CommunityStructure<uu::net::VertexLayerCommunity<const uu::net::Network>>>
to_communities(
               const py::dict& com,
               const uu::net::AttributedHomogeneousMultilayerNetwork* mnet
               )
{
    std::vector<std::string> cs_actor = com["actor"].cast<std::vector<std::string>>();;
    std::vector<std::string> cs_layer = com["layer"].cast<std::vector<std::string>>();;
    std::vector<size_t> cs_cid = com["cid"].cast<std::vector<size_t>>();;
    
    if ((cs_actor.size() != cs_layer.size()) || (cs_layer.size() != cs_cid.size()))
    {
        throw std::runtime_error("all lists should have the same length");
    }
    
    std::unordered_map<int, std::list<std::pair<const uu::net::Vertex*, const uu::net::Network*>> > result;
    
    for (size_t i=0; i<cs_actor.size(); i++) {
        int comm_id = cs_cid[i];
        auto layer = mnet->layers()->get(std::string(cs_layer[i]));
        if (!layer) throw std::runtime_error("cannot find layer " + std::string(cs_layer[i]) + " (community structure not compatible with this network?)");
        auto actor = mnet->vertices()->get(std::string(cs_actor[i]));
        if (!actor) throw std::runtime_error("cannot find actor " + std::string(cs_actor[i]) + " (community structure not compatible with this network?)");
        
        auto iv = std::make_pair(actor, layer);
        result[comm_id].push_back(iv);
        
    }
    
    
        // build community structure
    
    auto communities = std::make_unique<uu::net::CommunityStructure<uu::net::VertexLayerCommunity<const uu::net::Network>>>();
    
    for (auto pair: result)
    {
        auto c = std::make_unique<uu::net::VertexLayerCommunity<const uu::net::Network>>();
        
        for (auto vertex_layer_pair: pair.second)
        {
            c->add(vertex_layer_pair);
        }
        
        communities->add(std::move(c));
    }
    
    return communities;
}


