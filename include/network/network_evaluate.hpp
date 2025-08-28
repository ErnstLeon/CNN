#ifndef NETWORK_EVALUATE_HPP
#define NETWORK_EVALUATE_HPP

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "../core.hpp"
#include "network_base.hpp"

namespace CNN::Network
{

/*
    ! Not optimal yet !
    Propagates an input through the CNN and returns the output of the last layer.

    Function Arguments:
        input  : Input data for the CNN (HeapTensor3D with correct shape)

    Function Return value:
        output  : Output data of the CNN (HeapTensor1D with correct shape)

    Description:
        - The input is stored as the 0th post-activation output of the convolution layers.
        - From there, the input is sent through the network, where each layer takes the previous
          layer's post-activation output as input, computes its pre-activation and post-activation
          outputs, and stores them.
        - The final post-activation output of the convolution layers is stored flattened as
          the 0th post-activation output of the neural layers, and the process is
          repeated for the neural layers.
        - Return the last post-activation output of the neural layers
*/
template<
    typename Conv_Layer_Tuple, typename Neural_Layer_Tuple, 
    typename Conv_Feature_Tuple, typename Neural_Feature_Tuple>
requires(
    std::tuple_size_v<Conv_Layer_Tuple> == std::tuple_size_v<Conv_Feature_Tuple> - 1 && 
    std::tuple_size_v<Neural_Layer_Tuple> == std::tuple_size_v<Neural_Feature_Tuple> - 1)
HeapTensor1D<
    Network<Conv_Layer_Tuple, Neural_Layer_Tuple, Conv_Feature_Tuple, Neural_Feature_Tuple>::output_neurons,
    typename Network<Conv_Layer_Tuple, Neural_Layer_Tuple, Conv_Feature_Tuple, Neural_Feature_Tuple>::output_type> 
Network<
    Conv_Layer_Tuple, Neural_Layer_Tuple, 
    Conv_Feature_Tuple, Neural_Feature_Tuple>::evaluate(
    const HeapTensor3D<input_channels, input_height, input_width, input_type> & input)
{
    constexpr std::size_t Num_Conv_Layers = std::tuple_size_v<Conv_Layer_Tuple>;
    constexpr std::size_t Num_Neural_Layers = std::tuple_size_v<Neural_Layer_Tuple>;

    Conv_Feature_Tuple conv_weighted_inputs;
    Conv_Feature_Tuple conv_activation_results;
    Neural_Feature_Tuple neural_weighted_inputs;
    Neural_Feature_Tuple neural_activation_results;

    std::get<0>(conv_activation_results) = input;

    compile_range<Num_Conv_Layers>([&]<size_t I>(){
        std::get<I>(conv_layers).apply(std::get<I>(conv_activation_results), 
            std::get<I + 1>(conv_weighted_inputs), std::get<I + 1>(conv_activation_results));
    });

    std::get<0>(neural_activation_results) = 
        std::get<Num_Conv_Layers>(conv_activation_results);
    
    compile_range<Num_Neural_Layers>([&]<size_t I>(){
        std::get<I>(neural_layers).apply(std::get<I>(neural_activation_results), 
            std::get<I + 1>(neural_weighted_inputs), std::get<I + 1>(neural_activation_results));
    });

    return std::get<Num_Neural_Layers>(neural_activation_results);
}

}

#endif // NETWORK_EVALUATE_HPP