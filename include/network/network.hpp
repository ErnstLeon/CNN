#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "../core.hpp"
#include "activation.hpp"

namespace CNN::Network{

    template<typename Conv_Tuple, typename Neural_Tuple>
    class Network {
    private:

    Conv_Tuple conv_layers;
    Neural_Tuple neural_layers;

    static constexpr size_t num_conv_layers = std::tuple_size_v<Conv_Tuple>;
    static constexpr size_t num_neural_layers = std::tuple_size_v<Neural_Tuple>;
    
    public:

    Network(const Conv_Tuple& conv_layers, const Neural_Tuple& neural_layers) 
    : conv_layers{conv_layers},
    neural_layers{neural_layers} {};

    Network(Conv_Tuple&& conv_layers, Neural_Tuple&& neural_layers) 
    : conv_layers{std::move(conv_layers)},
    neural_layers{std::move(neural_layers)} {};

    
        
    };

    template<size_t CHANNELS, size_t IMG_HEIGHT, size_t IMG_WIDTH,
    size_t NUM_CONV_LAYERS, size_t NUM_NEURAL_LAYERS, typename... Layer_Args>
    auto network(Layer_Args&&... args){
        std::tuple<Layer_Args...> laysers{std::forward<Layer_Args>(args)...};

        auto conv_layers = slice_tuple<0, NUM_CONV_LAYERS - 1>(std::move(laysers));
        auto neural_layers = slice_tuple<NUM_CONV_LAYERS, NUM_CONV_LAYERS + NUM_NEURAL_LAYERS - 1>(std::move(laysers));

        auto conv_featureMaps = featureMaps_from_layer(conv_layers);

        return Network<decltype(conv_layers), decltype(neural_layers)>{ 
            std::move(conv_layers), std::move(neural_layers) 
        };

    }

}



#endif // NETWORK_HPP