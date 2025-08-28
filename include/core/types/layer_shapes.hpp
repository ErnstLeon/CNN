#ifndef LAYER_SHAPE_HPP
#define LAYER_SHAPE_HPP

#include <vector>
#include <random>

#include "heap_tensor.hpp"
#include "layer_types.hpp"
#include "stack_tensor.hpp"

namespace CNN
{

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

template<typename Layer_Tuple, size_t... Ids>
inline consteval bool output_eq_input_channels_helper(std::index_sequence<Ids...>) {
    return (... && (
        std::tuple_element_t<Ids, Layer_Tuple>::output_channels ==
        std::tuple_element_t<Ids + 1, Layer_Tuple>::input_channels));
}

template<typename Layer_Tuple>
inline consteval bool output_eq_input_channels() {
    constexpr size_t N = std::tuple_size_v<Layer_Tuple>;
    if constexpr (N < 2){
        return true;
    }
    else{
        return output_eq_input_channels_helper<Layer_Tuple>(std::make_index_sequence<N - 1>{});
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
inline constexpr auto features_from_layer_helper(std::index_sequence<Ids...>) {

    constexpr auto heights = get_heights<Layer_Tuple, H>();
    constexpr auto widths = get_widths<Layer_Tuple, W>();

    return std::tuple<
        HeapTensor3D<C, H, W>, 
        HeapTensor3D<std::tuple_element_t<Ids, Layer_Tuple>::output_channels, 
            heights[sizeof...(Ids) - 1 - Ids], widths[sizeof...(Ids) - 1 - Ids]>...>();
}

template<size_t C, size_t H, size_t W, typename Layer_Tuple>
inline constexpr auto features_from_layer() {
    constexpr size_t Num_featureMaps = std::tuple_size_v<Layer_Tuple>;
    return features_from_layer_helper<C, H, W, Layer_Tuple>(std::make_index_sequence<Num_featureMaps>{});
}

template<typename Layer_Tuple, size_t... Ids>
inline constexpr auto features_from_layer_helper(std::index_sequence<Ids...>) {

    static_assert(output_eq_input_neurons<Layer_Tuple>(), "Sizes of output and input neurons do not match"); 

    return std::tuple<
        HeapTensor1D<std::tuple_element_t<0, Layer_Tuple>::input_neurons>,
        HeapTensor1D<std::tuple_element_t<Ids, Layer_Tuple>::output_neurons>...>();
}

template<typename Layer_Tuple>
inline constexpr auto features_from_layer() {
    constexpr size_t Num_featureMaps = std::tuple_size_v<Layer_Tuple>;
    return features_from_layer_helper<Layer_Tuple>(std::make_index_sequence<Num_featureMaps>{});
}

template<typename Layer_Tuple,typename Optimizer, size_t... Ids>
inline constexpr auto weights_optimizer_helper(const Optimizer& optimizer, std::index_sequence<Ids...>) {
    
    return std::make_tuple(optimizer.template Tensor_Optimizer<
        decltype(std::get<Ids>(std::declval<Layer_Tuple>()).weights)>()...);
}

template<typename Layer_Tuple, typename Optimizer>
inline constexpr auto weights_optimizer(const Optimizer& optimizer) {
    constexpr size_t Num_Layers = std::tuple_size_v<Layer_Tuple>;
    return weights_optimizer_helper<Layer_Tuple>(optimizer, std::make_index_sequence<Num_Layers>{});
}

template<typename Layer_Tuple,typename Optimizer, size_t... Ids>
inline constexpr auto kernels_optimizer_helper(const Optimizer& optimizer, std::index_sequence<Ids...>) {
    
    return std::make_tuple(optimizer.template Tensor_Optimizer<
        decltype(std::get<Ids>(std::declval<Layer_Tuple>()).kernels)>()...);
}

template<typename Layer_Tuple, typename Optimizer>
inline constexpr auto kernels_optimizer(const Optimizer& optimizer) {
    constexpr size_t Num_Layers = std::tuple_size_v<Layer_Tuple>;
    return kernels_optimizer_helper<Layer_Tuple>(optimizer, std::make_index_sequence<Num_Layers>{});
}

template<typename Layer_Tuple,typename Optimizer, size_t... Ids>
inline constexpr auto biases_optimizer_helper(const Optimizer& optimizer, std::index_sequence<Ids...>) {
    
    return std::make_tuple(optimizer.template Tensor_Optimizer<
        decltype(std::get<Ids>(std::declval<Layer_Tuple>()).biases)>()...);
}

template<typename Layer_Tuple, typename Optimizer>
inline constexpr auto biases_optimizer(const Optimizer& optimizer) {
    constexpr size_t Num_Layers = std::tuple_size_v<Layer_Tuple>;
    return biases_optimizer_helper<Layer_Tuple>(optimizer, std::make_index_sequence<Num_Layers>{});
}

}

#endif // LAYER_SHAPE_HPP