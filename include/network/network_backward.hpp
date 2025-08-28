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

    std::get<Num_Neural_Layers - 1>(neural_layer_deriv).biases = 
        std::get<Num_Neural_Layers>(neural_activation_results);

    compile_range<std::tuple_element_t<Num_Neural_Layers - 1, Neural_Layer_Tuple>::output_neurons>(
    [&]<size_t J>()
    {
        std::get<Num_Neural_Layers - 1>(neural_layer_deriv).biases[J] -= true_output[J];
        
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
        std::get<Num_Neural_Layers - I>(neural_layers).apply_backwards(
            std::get<Num_Neural_Layers - I>(neural_layer_deriv).biases, 
            std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).biases);
        
        compile_range<std::tuple_element_t<Num_Neural_Layers - I - 1, Neural_Layer_Tuple>::output_neurons>(
        [&]<size_t J>()
        {
            std::get<Num_Neural_Layers - I - 1>(neural_layer_deriv).biases[J] *= 
            std::get<Num_Neural_Layers - I - 1>(neural_layers).activation_func.derivative(
                std::get<Num_Neural_Layers - I>(neural_weighted_inputs)[J]);
            
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

    Conv_Feature_Tuple layer_delta;

    // construct delta L for the convolution backward propagation
    std::get<0>(neural_layers).apply_backwards(
        std::get<0>(neural_layer_deriv).biases, 
        std::get<Num_Conv_Layers>(layer_delta));

    compile_range<std::tuple_element_t<0, Neural_Layer_Tuple>::input_neurons>(
    [&]<size_t I>()
    {
        std::get<Num_Conv_Layers>(layer_delta)[I] *= 
        std::get<Num_Conv_Layers - 1>(conv_layers).activation_func.derivative(
            std::get<Num_Conv_Layers>(conv_weighted_inputs)[I]);
    });

    // based on delta L, compule the kernel and bias derivatives for L
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

                    std::get<Num_Conv_Layers - 1>(conv_layer_deriv).kernels[kernel_deriv_id] = sum / (
                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size *
                        std::tuple_element_t<Num_Conv_Layers - 1, Conv_Layer_Tuple>::pooling_size);
                });
            });
        });
    });

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

        // based on delta L, compule the kernel and bias derivatives for L
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