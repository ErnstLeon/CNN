#ifndef NETWORK_GRADIENT_HPP
#define NETWORK_GRADIENT_HPP

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "../core.hpp"
#include "network_base.hpp"

namespace CNN::Network
{

/*
    Computes the gradients of the CNN with respect to its weights by performing a full 
    forward and backward pass.

    Function Arguments:
        input   : Input data for the CNN (HeapTensor3D with correct shape)
        output  : Target output vector (HeapTensor1D with correct size)
        conv_gradients   : Stores the computed gradients of the convolution layers
        neural_gradients : Stores the computed gradients of the fully-connected layers

    Description:
        - Initializes storage for pre-activation and post-activation outputs of both 
          convolution and neural layers.
        - Performs a forward pass with the input, storing all intermediate results 
          (weighted inputs and activations).
        - Performs a backward pass using the target output and the stored intermediate 
          results to compute the gradients of all layers.
*/
template<
    typename Conv_Layer_Tuple, typename Neural_Layer_Tuple, 
    typename Conv_Feature_Tuple, typename Neural_Feature_Tuple>
requires(
    std::tuple_size_v<Conv_Layer_Tuple> == std::tuple_size_v<Conv_Feature_Tuple> - 1 && 
    std::tuple_size_v<Neural_Layer_Tuple> == std::tuple_size_v<Neural_Feature_Tuple> - 1)
void Network<
    Conv_Layer_Tuple, Neural_Layer_Tuple, 
    Conv_Feature_Tuple, Neural_Feature_Tuple>::compute_gradient(
    const HeapTensor3D<input_channels, input_height, input_width, input_type> & input, 
    const HeapTensor1D<output_neurons, output_type> & output, 
    Conv_Layer_Tuple & conv_gradients,
    Neural_Layer_Tuple & neural_gradients)
{
    Conv_Feature_Tuple conv_weighted_inputs{};
    Conv_Feature_Tuple conv_activation_outputs{};
    Neural_Feature_Tuple neural_weighted_inputs{};
    Neural_Feature_Tuple neural_activation_outputs{};
    
    foward_propagate(input, 
        conv_weighted_inputs, 
        conv_activation_outputs, 
        neural_weighted_inputs, 
        neural_activation_outputs);

    backward_propagate(output, 
        conv_weighted_inputs, 
        conv_activation_outputs, 
        neural_weighted_inputs, 
        neural_activation_outputs,
        conv_gradients,
        neural_gradients);
}

}

#endif // NETWORK_GRADIENT_HPP