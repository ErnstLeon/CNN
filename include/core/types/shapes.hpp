#ifndef LAYER_SHAPE_HPP
#define LAYER_SHAPE_HPP

#include <vector>
#include <random>

#include "types.hpp"

namespace CNN
{

template<typename FeatureMap>
constexpr size_t flat_size = FeatureMap::channels * FeatureMap::height * FeatureMap::width;

template<typename Layer_Tuple, size_t... Ids>
inline consteval bool output_eq_input_neurons_helper(std::index_sequence<Ids...>) {
    return (... && (
        std::tuple_element_t<Ids, Layer_Tuple>::output_neurons ==
        std::tuple_element_t<Ids + 1, Layer_Tuple>::input_neurons));
}

template<typename Layer_Tuple>
inline consteval bool output_eq_input_neurons() {
    constexpr size_t N = std::tuple_size_v<Layer_Tuple>;
    if constexpr (N < 2){
        return true;
    }
    else{
        return output_eq_input_neurons_helper<Layer_Tuple>(std::make_index_sequence<N - 1>{});
    }
}

template<typename Layer_Tuple, size_t H, size_t... Hs>
inline consteval auto get_heights(){

    constexpr size_t Id = sizeof...(Hs);

    if constexpr (Id == std::tuple_size_v<Layer_Tuple>){
        return std::array<size_t, Id + 1>{H, Hs...};
    }
    else{
        constexpr size_t next_H = (((H + std::tuple_element_t<Id, Layer_Tuple>::stride - 1) / 
                std::tuple_element_t<Id, Layer_Tuple>::stride) + 
                std::tuple_element_t<Id, Layer_Tuple>::pooling_size - 1) / 
                std::tuple_element_t<Id, Layer_Tuple>::pooling_size;

        return get_heights<Layer_Tuple, next_H, H, Hs...>();
    }
}

template<typename Layer_Tuple, size_t W, size_t... Ws>
inline consteval auto get_widths(){

    constexpr size_t Id = sizeof...(Ws);

    if constexpr (Id == std::tuple_size_v<Layer_Tuple>){
        return std::array<size_t, Id + 1>{W, Ws...};
    }
    else{
        constexpr size_t next_W = (((W + std::tuple_element_t<Id, Layer_Tuple>::stride - 1) / 
                std::tuple_element_t<Id, Layer_Tuple>::stride) + 
                std::tuple_element_t<Id, Layer_Tuple>::pooling_size - 1) / 
                std::tuple_element_t<Id, Layer_Tuple>::pooling_size;

        return get_widths<Layer_Tuple, next_W, W, Ws...>();
    }
}

template<size_t C, size_t H, size_t W, typename Layer_Tuple, size_t... Ids>
inline constexpr auto featureMaps_from_layer_helper(const Layer_Tuple& layer_tuple, std::index_sequence<Ids...>) {

    constexpr auto heights = get_heights<Layer_Tuple, H>();
    constexpr auto widths = get_widths<Layer_Tuple, W>();

    return std::tuple<Convolution_FeatureMap<C, H, W>, Convolution_FeatureMap<
                std::tuple_element_t<Ids, Layer_Tuple>::output_channels, 
                heights[sizeof...(Ids) - 1 - Ids], widths[sizeof...(Ids) - 1 - Ids]>...>();
}

template<size_t C, size_t H, size_t W, typename Layer_Tuple>
inline constexpr auto featureMaps_from_layer(const Layer_Tuple& layer_tuple) {
    constexpr size_t Num_featureMaps = std::tuple_size_v<Layer_Tuple>;
    return featureMaps_from_layer_helper<C, H, W>(layer_tuple, std::make_index_sequence<Num_featureMaps>{});
}

template<typename Layer_Tuple, size_t... Ids>
inline constexpr auto featureMaps_from_layer_helper(const Layer_Tuple& layer_tuple, std::index_sequence<Ids...>) {

    static_assert(output_eq_input_neurons<Layer_Tuple>(), "Sizes of output and input neurons do not match"); 

    return std::tuple<Neural_FeatureMap<std::tuple_element_t<0, Layer_Tuple>::input_neurons>,
            Neural_FeatureMap<std::tuple_element_t<Ids, Layer_Tuple>::output_neurons>...>();
}

template<typename Layer_Tuple>
inline constexpr auto featureMaps_from_layer(const Layer_Tuple& layer_tuple) {
    constexpr size_t Num_featureMaps = std::tuple_size_v<Layer_Tuple>;
    return featureMaps_from_layer_helper(layer_tuple, std::make_index_sequence<Num_featureMaps>{});
}

}

#endif // LAYER_SHAPE_HPP