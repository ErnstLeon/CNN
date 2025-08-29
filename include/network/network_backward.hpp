#ifndef NETWORK_BACKWARD_HPP
#define NETWORK_BACKWARD_HPP

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "../core.hpp"
#include "network_base.hpp"

namespace CNN::Network
{

/*
    Performs backpropagation through the CNN, computing derivatives of weights and biases.

    Function Arguments:
        true_output              : True output (HeapTensor1D with correct shape).
        conv_weighted_inputs     : Pre-activation outputs of convolution + pooling layers.
        conv_activation_results  : Post-activation outputs of convolution + pooling layers.
        neural_weighted_inputs   : Pre-activation outputs of fully-connected layers.
        neural_activation_results: Post-activation outputs of fully-connected layers.
        conv_layer_deriv         : Stores computed gradients (kernels and biases) for convolutional layers.
        neural_layer_deriv       : Stores computed gradients (weights and biases) for fully-connected layers.

    Description:
        - Initializes the last fully-connected layer's deltas from the network output and the true labels.
        - Iteratively propagates deltas backwards through the fully-connected layers, applying activation derivatives
          and computing weight and bias gradients.
        - Propagates the error signal from the neural part into the convolutional stack.
        - For each convolutional layer (starting from the deepest):
            * Backpropagates pooled deltas to the previous feature maps.
            * Applies the activation function derivative elementwise.
            * Computes convolution kernel gradients by multiplying deltas with input activations.
            * Accumulates bias gradients as the sum of deltas.
        - Gradients are written into conv_layer_deriv and neural_layer_deriv, aligned with the
          network parameter structure.
*/
template<
    typename Conv_Layer_Tuple, 
    typename Neural_Layer_Tuple, 
    typename Conv_Feature_Tuple, 
    typename Neural_Feature_Tuple>
requires(
    std::tuple_size_v<Conv_Layer_Tuple> == std::tuple_size_v<Conv_Feature_Tuple> - 1 &&
    std::tuple_size_v<Neural_Layer_Tuple> == std::tuple_size_v<Neural_Feature_Tuple> - 1)
void Network<
    Conv_Layer_Tuple, Neural_Layer_Tuple, 
    Conv_Feature_Tuple, Neural_Feature_Tuple>::backward_propagate(
    const HeapTensor1D<output_neurons, output_type> & true_output, 
    const Conv_Feature_Tuple & conv_weighted_inputs, 
    const Conv_Feature_Tuple & conv_activation_results, 
    const Neural_Feature_Tuple & neural_weighted_inputs, 
    const Neural_Feature_Tuple & neural_activation_results,
    Conv_Layer_Tuple & conv_layer_deriv, 
    Neural_Layer_Tuple & neural_layer_deriv)
{
    constexpr std::size_t Num_Conv_Layers = std::tuple_size_v<Conv_Layer_Tuple>;
    constexpr std::size_t Num_Neural_Layers = std::tuple_size_v<Neural_Layer_Tuple>;

    // Initialize output layer delta for backpropagation
    // Assumes the last layer uses Softmax activation with Cross-Entropy loss
    // In this case, the derivative simplifies to (predicted_output - true_output)
    std::get<Num_Neural_Layers - 1>(neural_layer_deriv).biases = 
        std::get<Num_Neural_Layers>(neural_activation_results);

    compile_range<std::tuple_element_t<Num_Neural_Layers - 1, Neural_Layer_Tuple>::output_neurons>(
    [&]<size_t J>()
    {
        std::get<Num_Neural_Layers - 1>(neural_layer_deriv).biases[J] -= true_output[J];
        
        // Compute the gradient of the loss with respect to the last layer's weights.
        // For each weight connecting input neuron K to output neuron J, the gradient is:
        //     dL/dW[J,K] = delta[J] * activation_previous[K]
        compile_range<std::tuple_element_t<Num_Neural_Layers - 1, Neural_Layer_Tuple>::input_neurons>(
        [&]<size_t K>()
        {
            std::get<Num_Neural_Layers - 1>(neural_layer_deriv).weights[
                J * std::tuple_element_t<Num_Neural_Layers - 1, Neural_Layer_Tuple>::input_neurons + K] =
            std::get<Num_Neural_Layers - 1>(neural_layer_deriv).biases[J] *
            std::get<Num_Neural_Layers - 1>(neural_activation_results)[K];
        });
    });

    compile_range<Num_Neural_Layers, 1>(
    [&]<size_t I>()
    {
        // Apply the transposed weight matrix of the current layer to the current delta.
        // This propagates the error signal backward to the previous layer
        std::get<Num_Neural_Layers - I>(neural_layers).apply_backwards(
            std::get<Num_Neural_Layers - I>(neural_layer_deriv).biases, 
            std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).biases);
        
        // Multiply elementwise by the derivative of the activation function of the previous layer.
        // This gives the delta for the previous layer,
        compile_range<std::tuple_element_t<Num_Neural_Layers - I - 1, Neural_Layer_Tuple>::output_neurons>(
        [&]<size_t J>()
        {
            std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).biases[J] *= 
            std::get<Num_Neural_Layers - I - 1>(neural_layers).activation_func.derivative(
                std::get<Num_Neural_Layers - I>(neural_weighted_inputs)[J]);
            
            // Compute the gradient of the weights for the previous layer.
            // Each weight connecting neuron K to neuron J:
            //     dL/dW[J,K] = delta[J] * activation_previous[K]
            compile_range<std::tuple_element_t<Num_Neural_Layers - I - 1, Neural_Layer_Tuple>::input_neurons>(
            [&]<size_t K>()
            {
                std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).weights[
                    J * std::tuple_element_t<Num_Neural_Layers - I - 1, Neural_Layer_Tuple>::input_neurons + K] =
                std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).biases[J] *
                std::get<Num_Neural_Layers - I - 1>(neural_activation_results)[K];
            });
        });
    });

    // Initialize storage for convolutional layer deltas
    Conv_Feature_Tuple layer_delta;

    // Propagate the error from the first fully-connected (neural) layer
    // back into the last convolutional feature maps.
    // This applies the transposed weights of the first fully-connected layer to the output delta
    std::get<0>(neural_layers).apply_backwards(
        std::get<0>(neural_layer_deriv).biases, 
        std::get<Num_Conv_Layers>(layer_delta));

    // Apply elementwise multiplication by the derivative of the convolution layer's activation function.
    // This computes the delta for the last convolution layer's outputs, which serves as the starting point
    // for backpropagation through the convolutional layers.
    compile_range<std::tuple_element_t<0, Neural_Layer_Tuple>::input_neurons>(
    [&]<size_t I>()
    {
        std::get<Num_Conv_Layers>(layer_delta)[I] *= 
        std::get<Num_Conv_Layers - 1>(conv_layers).activation_func.derivative(
            std::get<Num_Conv_Layers>(conv_weighted_inputs)[I]);
    });

    // Compute gradients of the convolution kernels for the last convolutional layer
    // based on the delta of its output feature maps.
    //
    // Outer loops iterate over:
    //   OC = output channels (kernels) of the current conv layer
    //   IC = input channels feeding into each kernel
    //   KH, KW = kernel height and width indices
    //
    // For each spatial position in the delta map after pooling (OC, H, W),
    // iterate over the pooling window (PH, PW) to map the delta back
    // to the corresponding positions in the pre-pooled activation maps.
    //
    // Using stride, kernel offset, and pooling indices, compute the corresponding
    // activation position in the input feature maps (activation_height, activation_width).
    // Multiply this input activation by the delta at (OC, H, W) and sum contributions over all
    // positions in the pooling window to obtain the kernel gradient.
    //
    // Mathematically:
    //     dL/dK[OC, IC, KH, KW] = Σ_H Σ_W delta[OC, H, W] * ∂Z[OC, H, W] / ∂K[OC, IC, KH, KW]
    // where ∂Z/∂K represents the input activation corresponding to the kernel element.
    //
    // Effectively, each kernel gradient element is the sum of deltas multiplied by the input values
    // that contributed to the respective output positions, weighted by the kernel.
    compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::output_channels>(
    [&]<size_t OC>()
    {
        compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::input_channels>(
        [&]<size_t IC>()
        {
            compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size>(
            [&]<size_t KH>()
            {
                compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size>(
                [&]<size_t KW>()
                {
                    typename std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::type sum = 0;

                    compile_range<std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_y>(
                    [&]<size_t H>()
                    {
                        compile_range<std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z>(
                        [&]<size_t W>()
                        {   
                            constexpr size_t delta_id = 
                                OC * std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_y *
                                std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z +
                                H * std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z +
                                W;

                            constexpr size_t pool_h = H * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size;
                            constexpr size_t pool_w = W * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size;

                            compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size>(
                            [&]<size_t PW>()
                            {
                                compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size>(
                                [&]<size_t PH>()
                                {
                                    constexpr size_t activation_height = 
                                        (pool_h + PH) * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::stride - 
                                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size / 2 + KH;

                                    constexpr size_t activation_width = 
                                        (pool_w + PW) * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::stride - 
                                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size / 2 + KW;

                                    if constexpr (activation_height >= 0 && activation_width >= 0 &&
                                        activation_height < std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::size_y &&
                                        activation_width < std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::size_z)
                                    {
                                        constexpr size_t activation_id = 
                                            IC * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::size_y *
                                            std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::size_z +
                                            activation_height * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Feature_Tuple>::size_z +
                                            activation_width;

                                        sum += std::get<Num_Conv_Layers>(layer_delta)[delta_id] * 
                                            std::get<Num_Conv_Layers - 1>(conv_activation_results)[activation_id];
                                    }
                                });
                            });
                        });
                    });
                    
                    constexpr size_t kernel_deriv_id = 
                        OC * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::input_channels *
                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size *
                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size +
                        IC * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size *
                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size +
                        KH * std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::kernel_size + 
                        KW;

                    // If the convolution layer uses average pooling, divide by the pooling window area
                    // to account for the averaging effect.
                    std::get<Num_Conv_Layers - 1>(conv_layer_deriv).kernels[kernel_deriv_id] = sum / (
                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size *
                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size);
                });
            });
        });
    });

    // Compute the gradient of the biases for the last convolutional layer.
    // For each output channel (OC), sum the delta values over all spatial positions (H, W)
    // in the corresponding feature map. This gives:
    //     dL/db[OC] = Σ_H Σ_W delta[OC, H, W]
    compile_range<std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::output_channels>(
    [&]<size_t OC>()
    {   
        typename std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::type sum = 0;

        compile_range<std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_y>(
        [&]<size_t H>()
        {
            compile_range<std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z>(
            [&]<size_t W>()
            {   
                constexpr size_t delta_id = 
                    OC * std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_y *
                    std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z +
                    H * std::tuple_element_t<Num_Conv_Layers, Conv_Feature_Tuple>::size_z +
                    W;
                
                sum += std::get<Num_Conv_Layers>(layer_delta)[delta_id];
            });
        });

        std::get<Num_Conv_Layers - 1>(conv_layer_deriv).biases[OC] = sum;
    });

    // Backpropagation through each convolutional layer:
    //
    // Propagate the delta from the next layer to the current layer using the layer's backward function.
    //       delta_current = layer.apply_backwards(delta_next)
    //
    // Multiply elementwise by the derivative of the activation function.
    //       delta_current[i] *= f'(z_current[i])
    //
    // After this step, layer_delta contains the correctly scaled deltas for the current layer,
    // which are then used in the previously described kernel and bias gradient computations.
    compile_range<Num_Conv_Layers, 1>(
    [&]<size_t I>()
    {
        std::get<Num_Conv_Layers - I>(conv_layers).apply_backwards(
            std::get<Num_Conv_Layers - I + 1>(layer_delta), 
            std::get<Num_Conv_Layers - I>(layer_delta));

        CNN::compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::output_channels>(
        [&]<size_t OC>()
        {
            compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_y>(
            [&]<size_t H>()
            {
                compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z>(
                [&]<size_t W>()
                {   
                    constexpr size_t delta_id = 
                        OC * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_y *
                        std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z + 
                        H * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z + W;

                    std::get<Num_Conv_Layers - I>(layer_delta)[delta_id] *= 
                        std::get<Num_Conv_Layers - I - 1>(conv_layers).activation_func.derivative(
                        std::get<Num_Conv_Layers - I>(conv_weighted_inputs)[delta_id]);
                });
            });
        });

        // As described above, use the current layer's delta to compute the kernel and bias gradients.
        compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::output_channels>(
        [&]<size_t OC>()
        {
            compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::input_channels>(
            [&]<size_t IC>()
            {
                compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size>(
                [&]<size_t KH>()
                {
                    compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size>(
                    [&]<size_t KW>()
                    {
                        typename std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::type sum = 0;

                        compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_y>(
                        [&]<size_t H>()
                        {
                            compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z>(
                            [&]<size_t W>()
                            {   
                                constexpr size_t delta_id = 
                                    OC * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_y *
                                    std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z +
                                    H * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z +
                                    W;

                                constexpr size_t pool_h = H * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::pooling_size;
                                constexpr size_t pool_w = W * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::pooling_size;

                                compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::pooling_size>(
                                [&]<size_t PW>()
                                {
                                    compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::pooling_size>(
                                    [&]<size_t PH>()
                                    {
                                        constexpr size_t activation_height = 
                                            (pool_h + PH) * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::stride - 
                                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size / 2 + KH;

                                        constexpr size_t activation_width = 
                                            (pool_w + PW) * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::stride - 
                                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size / 2 + KW;

                                        if constexpr (activation_height >= 0 && activation_width >= 0 &&
                                            activation_height < std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::size_y &&
                                            activation_width < std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::size_z)
                                        {
                                            constexpr size_t activation_id = 
                                                IC * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::size_y *
                                                std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::size_z +
                                                activation_height * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Feature_Tuple>::size_z +
                                                activation_width;

                                            sum += std::get<Num_Conv_Layers - I>(layer_delta)[delta_id] * 
                                                std::get<Num_Conv_Layers - I - 1>(conv_activation_results)[activation_id];
                                        }
                                    });
                                });
                            });
                        });
                        
                        constexpr size_t kernel_deriv_id = 
                            OC * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::input_channels *
                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size *
                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size +
                            IC * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size *
                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size +
                            KH * std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::kernel_size + 
                            KW;

                        std::get<Num_Conv_Layers - I - 1>(conv_layer_deriv).kernels[kernel_deriv_id] = sum /(
                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::pooling_size *
                            std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::pooling_size);
                    });
                });
            });
        });

        compile_range<std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::output_channels>(
        [&]<size_t OC>()
        {   
            typename std::tuple_element_t<Num_Conv_Layers - I - 1, Conv_Layer_Tuple>::type sum = 0;

            compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_y>(
            [&]<size_t H>()
            {
                compile_range<std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z>(
                [&]<size_t W>()
                {   
                    constexpr size_t delta_id = 
                        OC * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_y *
                        std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z + 
                        H * std::tuple_element_t<Num_Conv_Layers - I, Conv_Feature_Tuple>::size_z +
                        W;
                    
                    sum += std::get<Num_Conv_Layers - I>(layer_delta)[delta_id];
                });
            });

            std::get<Num_Conv_Layers - I - 1>(conv_layer_deriv).biases[OC] = sum;
        });
    });
}

}

#endif // NETWORK_BACKWARD_HPP