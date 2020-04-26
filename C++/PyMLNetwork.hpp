
#ifndef UU_MULTINET_PYMLNETWORK_H_
#define UU_MULTINET_PYMLNETWORK_H_

#include "networks/AttributedHomogeneousMultilayerNetwork.hpp"

class PyMLNetwork
{
private:
    std::shared_ptr<uu::net::AttributedHomogeneousMultilayerNetwork> ptr;
    
public:
    
    std::string
    name(
    ) const
    {
        return ptr->name;
    }
    
    PyMLNetwork(std::shared_ptr<uu::net::AttributedHomogeneousMultilayerNetwork> ptr) : ptr(ptr)
    {
            // @todo check not null?
    }
    
    uu::net::AttributedHomogeneousMultilayerNetwork*
    get_mlnet() const
    {
        return ptr.get();
    }
    
};

#endif
