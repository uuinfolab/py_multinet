
#ifndef UU_MULTINET_PYMLNETWORK_H_
#define UU_MULTINET_PYMLNETWORK_H_

#include "networks/MultilayerNetwork.hpp"

class PyMLNetwork
{
private:
    std::shared_ptr<uu::net::MultilayerNetwork> ptr;
    
public:
    
    std::string
    name(
    ) const
    {
        return ptr->name;
    }
    
    PyMLNetwork(std::shared_ptr<uu::net::MultilayerNetwork> ptr) : ptr(ptr)
    {
            // @todo check not null?
    }
    
    uu::net::MultilayerNetwork*
    get_mlnet() const
    {
        return ptr.get();
    }
    
};

#endif
