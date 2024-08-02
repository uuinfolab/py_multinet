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
#include "measures/degree_ml.hpp"
#include "measures/relevance.hpp"
#include "measures/redundancy.hpp"
#include "measures/layer.hpp"
#include "measures/distance.hpp"
#include "core/propertymatrix/summarization.hpp"
#include "layout/multiforce.hpp"
#include "layout/circular.hpp"


using M = uu::net::MultilayerNetwork;
using G = uu::net::Network;





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

