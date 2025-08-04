#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "core/compile_range.hpp"
#include "core/network_types.hpp"

namespace CNN::Network{

    template<typename T, size_t NUM_CONV_LAYERS, size_t NUM_NEURAL_LAYERS>
    class Network {
    private:
    
    public:
        
      /*  template<typename... Layer_Args>
        requires (sizeof...(Layer_Args) == NUM_CONV_LAYERS + NUM_NEURAL_LAYERS)
        Network(Layer_Args&&... layers) {
            
            std::tuple<Layer_Args...> layers_{std::forward<Layer_Args>(layers)...};
            
            compile_range<NUM_CONV_LAYERS - 1>([&]<size_t I>(){
               conv_layers[I] = std::move(std::get<I>(layers_));
            });

            compile_range<NUM_CONV_LAYERS + NUM_NEURAL_LAYERS - 1, NUM_CONV_LAYERS>([&]<size_t I>(){
               neural_layers[I] = std::move(std::get<I>(layers_));
            });
            
        }
*/


    };

}



#endif // NETWORK_HPP