#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "../core.hpp"
#include "activation.hpp"

namespace CNN::Network
{

template<typename Conv_Layer_Tuple, typename Neural_Layer_Tuple, 
typename Conv_Feature_Tuple, typename Neural_Feature_Tuple>
class Network {
private:

    Conv_Layer_Tuple conv_layers;
    Neural_Layer_Tuple neural_layers;

    Conv_Feature_Tuple conv_features;
    Neural_Feature_Tuple neural_features;

    static constexpr size_t num_conv_layers = std::tuple_size_v<Conv_Layer_Tuple>;
    static constexpr size_t num_neural_layers = std::tuple_size_v<Neural_Layer_Tuple>;

public:

    Network(const Conv_Layer_Tuple& conv_layers, const Neural_Layer_Tuple& neural_layers,
        const Conv_Feature_Tuple& conv_features, const Neural_Feature_Tuple& neural_features) 
    : conv_layers{conv_layers},
    neural_layers{neural_layers},
    conv_features{conv_features},
    neural_features{neural_features} {};

    Network(Conv_Layer_Tuple&& conv_layers, Neural_Layer_Tuple&& neural_layers,
        Conv_Feature_Tuple&& conv_features, Neural_Feature_Tuple&& neural_features) 
    : conv_layers{std::move(conv_layers)},
    neural_layers{std::move(neural_layers)},
    conv_features{std::move(conv_features)},
    neural_features{std::move(neural_features)} {};

};

template<size_t C, size_t H, size_t W,
size_t NUM_CONV_LAYERS, size_t NUM_NEURAL_LAYERS, typename... Layer_Args>
requires (NUM_CONV_LAYERS + NUM_NEURAL_LAYERS == sizeof...(Layer_Args))
inline auto network(Layer_Args&&... args){

    auto [conv_layers, neural_layers] = split_tuple<NUM_CONV_LAYERS - 1>(
        std::tuple<Layer_Args...>{std::forward<Layer_Args>(args)...});

    auto conv_featureMaps = featureMaps_from_layer<C, H, W>(conv_layers);

    using Conv_Layers_t = decltype(conv_layers);
    using Conv_FeatureMap_t = decltype(conv_featureMaps);
    using Neural_Layers_t = decltype(neural_layers);
    using LastConvFeatureMap = std::tuple_element_t<NUM_CONV_LAYERS, Conv_FeatureMap_t>;
    using FirstNeuralLayer = std::tuple_element_t<0, Neural_Layers_t>;

    if constexpr (flat_size<LastConvFeatureMap> != FirstNeuralLayer::input_neurons)
    {
        auto extended_neural_layers = add_tuple_begin(
            std::move(neural_layers), Neural_Layer<flat_size<LastConvFeatureMap>, 
            FirstNeuralLayer::input_neurons, typename FirstNeuralLayer::type>{});
            
        auto extended_neural_featureMaps = featureMaps_from_layer(extended_neural_layers);

        return Network<Conv_Layers_t, decltype(extended_neural_layers),
            decltype(conv_featureMaps), decltype(extended_neural_featureMaps)>{ 
            std::move(conv_layers), std::move(extended_neural_layers),
            std::move(conv_featureMaps), std::move(extended_neural_featureMaps)};
    }
    else
    {
        auto neural_featureMaps = featureMaps_from_layer(neural_layers);

        return Network<Conv_Layers_t, Neural_Layers_t,
            Conv_FeatureMap_t, decltype(neural_featureMaps)>{ 
            std::move(conv_layers), std::move(neural_layers),
            std::move(conv_featureMaps), std::move(neural_featureMaps)};
    }
}

}



#endif // NETWORK_HPP